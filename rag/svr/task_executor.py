#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime
import json
import logging
import os
import hashlib
import copy
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from multiprocessing.context import TimeoutError
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from elasticsearch_dsl import Q

from api.db import LLMType, ParserType
from api.db.services.dialog_service import keyword_extraction, question_proposal
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.db.services.task_service import TaskService
from api.db.services.file2document_service import File2DocumentService
from api.settings import retrievaler
from api.utils.file_utils import get_project_base_directory
from api.db.db_models import close_connection
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, knowledge_graph, email
from rag.nlp import search, rag_tokenizer
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.settings import database_logger, SVR_QUEUE_NAME
from rag.settings import cron_logger, DOC_MAXIMUM_SIZE
from rag.utils import rmSpace, num_tokens_from_string
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils.redis_conn import REDIS_CONN, Payload
from rag.utils.storage_factory import STORAGE_IMPL

BATCH_SIZE = 64

# 预先导入了各种类型的文件解析/切块器(chunker)
FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: knowledge_graph
}


# redis消费者名称，用于区分不同的消费者,在这里用进程编号(参数1)作为消费者名称
CONSUMER_NAME = "task_consumer_" + ("0" if len(sys.argv) < 2 else sys.argv[1])
PAYLOAD: Payload | None = None


def set_progress(task_id, from_page=0, to_page=-1, prog=None, msg="Processing..."):
    global PAYLOAD
    if prog is not None and prog < 0:
        msg = "[ERROR]" + msg
    cancel = TaskService.do_cancel(task_id)
    if cancel:
        msg += " [Canceled]"
        prog = -1

    if to_page > 0:
        if msg:
            msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
    d = {"progress_msg": msg}
    if prog is not None:
        d["progress"] = prog
    try:
        TaskService.update_progress(task_id, d)
    except Exception as e:
        cron_logger.error("set_progress:({}), {}".format(task_id, str(e)))

    close_connection()
    if cancel:
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
        os._exit(0)


def collect():
    # 声明全局变量，以便可以在函数内部修改它们
    global CONSUMER_NAME, PAYLOAD

    try:
        # 尝试从Redis队列中获取未确认的消息（就是尚未完成的任务）,在消息队列系统中，消息通常经过以下几个阶段：发布\获取\处理\确认\重试
        PAYLOAD = REDIS_CONN.get_unacked_for(CONSUMER_NAME, SVR_QUEUE_NAME, "rag_flow_svr_task_broker")

        # 如果没有未确认的消息，则作为消费者从队列中获取一条消息
        if not PAYLOAD:
            PAYLOAD = REDIS_CONN.queue_consumer(SVR_QUEUE_NAME, "rag_flow_svr_task_broker", CONSUMER_NAME)

        # 如果还是没有消息，则等待一秒后返回空DataFrame
        if not PAYLOAD:
            time.sleep(1)
            return pd.DataFrame()

    except Exception as e:
        # 记录获取队列事件时发生的异常,返回空的DataFrame
        cron_logger.error("Get task event from queue exception:" + str(e))
        return pd.DataFrame()

    # 提取消息内容
    msg = PAYLOAD.get_message()

    # 如果消息为空，则返回空DataFrame
    if not msg:
        return pd.DataFrame()

    # 检查任务是否已被取消,返回空的DataFrame
    if TaskService.do_cancel(msg["id"]):
        # 记录任务已取消的日志
        cron_logger.info("Task {} has been canceled.".format(msg["id"]))
        return pd.DataFrame()

    # 获取与消息ID相关联的任务列表，任务列表的任务包括：文档切割、文档嵌入等
    tasks = TaskService.get_tasks(msg["id"])

    # 如果任务列表为空，则记录警告日志并返回空列表
    if not tasks:
        cron_logger.warn("{} empty task!".format(msg["id"]))
        return []

    # 将任务列表转换为pandas DataFrame
    tasks = pd.DataFrame(tasks)

    # 如果消息类型为 "raptor"，则添加一个新的列 "task_type" 并设置其值为 "raptor"
    if msg.get("type", "") == "raptor":
        tasks["task_type"] = "raptor"

    # 返回包含任务数据的DataFrame
    return tasks


def get_storage_binary(bucket, name):
    return STORAGE_IMPL.get(bucket, name)


def build(row):
    """
    对文件切块，自动生成关键词，自动生成QA，然后把这些信息放到到docs，并返回
    row: 任务配置- doc_id \ location\ name \ size \ parser_id切块器id \ parser_config
    返回 docs: doc_id \ kb_id \
    """

    # 检查文件大小是否超过限制
    if row["size"] > DOC_MAXIMUM_SIZE:
        set_progress(row["id"], prog=-1, msg="File size exceeds( <= %dMb )" %
                                             (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
        return []

    # 创建一个回调函数，用于更新任务进度
    callback = partial(
        set_progress,
        row["id"],
        row["from_page"],
        row["to_page"])

    # 获取对应的切块器
    chunker = FACTORY[row["parser_id"].lower()]

    try:
        # 获取文件的存储地址
        st = timer()
        bucket, name = File2DocumentService.get_storage_address(doc_id=row["doc_id"])
        # 从minio存储中获取文件二进制内容
        binary = get_storage_binary(bucket, name)
        cron_logger.info(
            "From minio({}) {}/{}".format(timer() - st, row["location"], row["name"]))
    except TimeoutError:
        callback(-1, "Internal server error: Fetch file from minio timeout. Could you try it again.")
        cron_logger.error(
            "Minio {}/{}: Fetch file from minio timeout.".format(row["location"], row["name"]))
        return
    except Exception as e:
        if re.search("(No such file|not found)", str(e)):
            callback(-1, "Can not find file <%s> from minio. Could you try it again?" % row["name"])
        else:
            callback(-1, "Get file from minio: %s" % str(e).replace("'", ""))
        traceback.print_exc()
        return
    
    try:
        # 使用切块器对文件进行解析、切块、合并较小的块、tokenize
        # naive.chunk(row["name"], binary=binary, from_page=row["from_page"],
        #                     to_page=row["to_page"], lang=row["language"], callback=callback,
        #                     kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        
        cks = chunker.chunk(row["name"], binary=binary, from_page=row["from_page"],
                            to_page=row["to_page"], lang=row["language"], callback=callback,
                            kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        
        cron_logger.info(
            "Chunking({}) {}/{}".format(timer() - st, row["location"], row["name"]))
    except Exception as e:
        callback(-1, "Internal server error while chunking: %s" %
                     str(e).replace("'", ""))
        cron_logger.error(
            "Chunking {}/{}: {}".format(row["location"], row["name"], str(e)))
        traceback.print_exc()
        return


    # 初始化文档列表(在这里文件的一个切块被视为一个doc)
    docs = []
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])]
    }

    # 计时变量
    el = 0

    # 遍历切块结果
    for ck in cks:
        # 深拷贝doc对象，深拷贝意味着新对象完全独立于原对象
        d = copy.deepcopy(doc)

        # 字典 ck 中的所有键值对合并到字典 d 中。如果 ck 中的键已经在 d 中存在，那么 d 中对应的键的值将被 ck 中的值覆盖。
        d.update(ck)

        # 为切块信息生成唯一标识符
        md5 = hashlib.md5()
        md5.update((ck["content_with_weight"] +
                    str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()

        # 为切块信息生成时间信息
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()

        # 如果切块没有图像数据,则跳过下面的处理
        if not d.get("image"):
            docs.append(d)
            continue

        # 处理文档切块中的图像数据，并将其存储到指定的存储实现中
        try:
            # 初始化缓冲区
            output_buffer = BytesIO()

            # 检查 d["image"] 是否为字节串类型 (bytes)。
            # 如果是字节串类型，直接使用 BytesIO 将其包装成一个文件对象。
            # 如果不是字节串类型，假设 d["image"] 是一个支持 .save 方法的对象（如PIL图像对象），将其保存到 output_buffer 中，格式为 JPEG。
            if isinstance(d["image"], bytes):
                output_buffer = BytesIO(d["image"])
            else:
                d["image"].save(output_buffer, format='JPEG')

            st = timer()

            # 从 output_buffer 中获取图像数据的字节串表示。
            # 调用 STORAGE_IMPL.put 方法，将图像数据存储到指定的存储实现中（在这里是MinIO）。
            # row["kb_id"] 是知识库的 ID。
            # d["_id"] 是文档切块的唯一标识符。
            # output_buffer.getvalue() 是图像数据的字节串表示。
            STORAGE_IMPL.put(row["kb_id"], d["_id"], output_buffer.getvalue())
            el += timer() - st
        except Exception as e:
            cron_logger.error(str(e))
            traceback.print_exc()

        d["img_id"] = "{}-{}".format(row["kb_id"], d["_id"])
        del d["image"]
        docs.append(d)
    cron_logger.info("MINIO PUT({}):{}".format(row["name"], el))

    # 如果配置中有自动关键词生成，则生成关键词
    if row["parser_config"].get("auto_keywords", 0):
        callback(msg="Start to generate keywords for every chunk ...")
        chat_mdl = LLMBundle(row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"])
        for d in docs:
            d["important_kwd"] = keyword_extraction(chat_mdl, d["content_with_weight"],
                                                    row["parser_config"]["auto_keywords"]).split(",")
            d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))


    # 如果配置中有自动问题生成，则生成问题
    if row["parser_config"].get("auto_questions", 0):
        callback(msg="Start to generate questions for every chunk ...")
        chat_mdl = LLMBundle(row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"])
        for d in docs:
            qst = question_proposal(chat_mdl, d["content_with_weight"], row["parser_config"]["auto_questions"])
            d["content_with_weight"] = f"Question: \n{qst}\n\nAnswer:\n" + d["content_with_weight"]
            qst = rag_tokenizer.tokenize(qst)
            if "content_ltks" in d:
                d["content_ltks"] += " " + qst
            if "content_sm_ltks" in d:
                d["content_sm_ltks"] += " " + rag_tokenizer.fine_grained_tokenize(qst)

    # 返回最终处理后的文档列表
    return docs


def init_kb(row):
    """
    Elasticsearch 中为特定租户初始化一个知识库索引，如果索引已经存在则不做任何操作，否则创建索引并应用指定的映射配置。这
    """
    idxnm = search.index_name(row["tenant_id"])
    if ELASTICSEARCH.indexExist(idxnm):
        return
    return ELASTICSEARCH.createIdx(idxnm, json.load(
        open(os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r")))


def embedding(docs, mdl, parser_config=None, callback=None):
    """
    对文档标题和内容进行编码，并根据一定的权重组合这两个嵌入向量。
    docs: 文档列表
    mdl: 模型
    parser_config: 解析器配置
    callback: 回调函数
    """

    if parser_config is None:
        parser_config = {}

    batch_size = 32

    # 准备标题和内容，tts是标题 cnts是内容
    # tts：这是一个列表推导式，遍历 docs 列表中的每一个文档 d。
    # 使用 d.get("title_tks") 检查文档是否有标题词元。
    # 如果文档有标题词元，则调用 rmSpace 函数来移除词元中的空白字符。
    # 将处理后的标题词元加入到 tts 列表中。

    # cnts：这也是一个列表推导式，同样遍历 docs 列表中的每一个文档 d。
    # 使用正则表达式 re.sub 来替换文档中的 HTML 表格标签。
    # 正则表达式 r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>" 匹配所有的表格相关标签，包括 <table>, <td>, <caption>, <tr>, <th> 及其关闭标签，并且允许标签内有最多 12 个非尖括号字符的属性。
    # 替换匹配到的标签为单个空格 " "。
    # 将清理后的文档内容文本加入到 cnts 列表中    
    tts, cnts = [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")], [
        re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", d["content_with_weight"]) for d in docs]


    # 处理标题嵌入
    tk_count = 0
    # 初始化一个空的 NumPy 数组 tts_，用于存放处理后的标题嵌入向量。
    if len(tts) == len(cnts):
        tts_ = np.array([])

        # 使用列表推导式遍历 tts，以 batch_size 为步长进行批量处理。
        # 对于每个批次的数据 tts[i : i + batch_size]，调用 mdl.encode() 方法进行嵌入处理，返回嵌入向量 vts 和词汇数量 c。
        # 如果 tts_ 数组是空的，则直接赋值为 vts；否则，将 vts 与现有的 tts_ 数组沿轴 0 方向进行拼接。
        # 累加处理得到的词汇数量 c 到 tk_count。
        # 调用 callback 函数更新进度条，进度条的值从 0.6 开始，逐渐增加到 0.7，这表示标题嵌入处理阶段的完成情况。
        for i in range(0, len(tts), batch_size):
            vts, c = mdl.encode(tts[i: i + batch_size])
            if len(tts_) == 0:
                tts_ = vts
            else:
                tts_ = np.concatenate((tts_, vts), axis=0)
            tk_count += c
            callback(prog=0.6 + 0.1 * (i + 1) / len(tts), msg="")
        tts = tts_

    # 处理内容嵌入,解释同上
    cnts_ = np.array([])
    for i in range(0, len(cnts), batch_size):
        vts, c = mdl.encode(cnts[i: i + batch_size])
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    # 计算标题权重,取parser_config["filename_embd_weight"]值。如果没有设置该键，则默认使用 0.1。
    title_w = float(parser_config.get("filename_embd_weight", 0.1))

    # 如果标题嵌入向量 tts 和内容嵌入向量 cnts 的长度相同，那么就按照权重 title_w 组合这两个向量。
    # 如果长度不同，则直接使用内容嵌入向量 cnts 作为最终的文档嵌入向量
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts

    # 检查向量数量是否与文档数量一致
    assert len(vects) == len(docs)

    # 将文档嵌入向量存储到文档docs中
    for i, d in enumerate(docs):
        # 从嵌入向量数组 vects 中获取第 i 个文档的嵌入向量。使用 .tolist() 方法将 NumPy 数组转换为 Python 列表。
        v = vects[i].tolist()
        # 键名格式为 "q_<嵌入向量长度>_vec"，例如，如果嵌入向量的长度为 768，则键名为 "q_768_vec"。
        d["q_%d_vec" % len(v)] = v

    return tk_count


# 对文档进行聚类,对每个类生成摘要，对摘要进行嵌入，将摘要内容和嵌入向量添加到chunks中
def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    vts, _ = embd_mdl.encode(["ok"])
    vctr_nm = "q_%d_vec" % len(vts[0])
    chunks = []

    # 从es中获取文档的内容和嵌入向量，并将它们添加到 chunks 列表中。
    for d in retrievaler.chunk_list(row["doc_id"], row["tenant_id"], fields=["content_with_weight", vctr_nm]):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    # 它使用高斯混合模型 (Gaussian Mixture Model, GMM) 对文档嵌入进行聚类(簇)，并对每个簇生成摘要。
    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    original_length = len(chunks)
    # 进行聚类、摘要生成和摘要嵌入生成，并附加到chunks中
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)

    # 深拷贝chunks,然后更新chunks项目中的关键字段，结果存放到res
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    res = []
    tk_count = 0
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    return res, tk_count


def main():
    # 从Redis收集一个文档的任务列表
    rows = collect()
    # 如果没有任务，则直接返回
    if len(rows) == 0:
        return

    for _, r in rows.iterrows():
        # 创建一个回调函数，用于更新任务进度
        callback = partial(set_progress, r["id"], r["from_page"], r["to_page"])

        try:
            # 创建一个嵌入模型实例
            embd_mdl = LLMBundle(r["tenant_id"], LLMType.EMBEDDING, llm_name=r["embd_id"], lang=r["language"])
        except Exception as e:
            # 如果创建嵌入模型失败，则记录错误并继续下一个任务
            callback(-1, msg=str(e))
            cron_logger.error(str(e))
            continue

        if r.get("task_type", "") == "raptor":
            try:
                # 如果任务类型是 "raptor"，则创建一个聊天模型实例，用于生成文件的摘要
                chat_mdl = LLMBundle(r["tenant_id"], LLMType.CHAT, llm_name=r["llm_id"], lang=r["language"])
                # 执行文档聚簇和摘要生成任务,cks是文档原来的chunks(从ES搜索得来)添加了摘要和摘要嵌入以后，生成的新的chunks
                cks, tk_count = run_raptor(r, chat_mdl, embd_mdl, callback)
            except Exception as e:
                # 如果 Raptor 任务执行失败，则记录错误并继续下一个任务
                callback(-1, msg=str(e))
                cron_logger.error(str(e))
                continue
        else:
            # 对文件切块，自动生成关键词，自动生成QA，然后把这些信息放到到cks数组中（一个切块一条记录）
            st = timer()

            cks = build(r)
            # 记录构建耗时
            cron_logger.info("Build chunks({}): {}".format(r["name"], timer() - st))

            # 如果chunks为 None，则跳过后续处理
            if cks is None:
                continue

            if not cks:
                # 如果没有构建任何 chunk，则记录信息并继续下一个任务
                callback(1., "No chunk! Done!")
                continue

            # 提示文件切块完成
            callback(
                msg="Finished slicing files(%d). Start to embedding the content." %
                    len(cks))
            # 开始计时
            st = timer()
            try:
                # 进行文件嵌入,返回后cks增加了新的键值对（例如q_768_vec），包含了文档嵌入向量
                tk_count = embedding(cks, embd_mdl, r["parser_config"], callback)
            except Exception as e:
                # 如果嵌入过程出错，则记录错误并继续下一个任务
                callback(-1, "Embedding error:{}".format(str(e)))
                cron_logger.error(str(e))
                tk_count = 0
            # 记录嵌入耗时
            cron_logger.info("Embedding elapsed({}): {:.2f}".format(r["name"], timer() - st))
            # 设置进度
            callback(msg="Finished embedding({:.2f})! Start to build index!".format(timer() - st))

        # 上面已经得到了cks(chunks),下面是保存到es中
        # 在ES中为该知识库新生成索引，如果索引已经存在则退出,知识库的所有文件的所有文档切块的嵌入向量都由这个索引进行索引
        init_kb(r)

        # 计算 chunk 数量
        chunk_count = len(set([c["_id"] for c in cks]))
        # 开始计时
        st = timer()
        es_r = ""
        es_bulk_size = 4
        # 分批插入 Elasticsearch
        for b in range(0, len(cks), es_bulk_size):
            es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]))
            if b % 128 == 0:
                # 更新进度
                callback(prog=0.8 + 0.1 * (b + 1) / len(cks), msg="")

        # 记录索引耗时
        cron_logger.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
        if es_r:
            # 如果 Elasticsearch 插入失败，则记录错误并删除相关数据
            callback(-1, "Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!")
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
            cron_logger.error(str(es_r))
        else:
            if TaskService.do_cancel(r["id"]):
                # 如果任务被取消，则删除相关数据
                ELASTICSEARCH.deleteByQuery(
                    Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"]))
                continue
            # 任务完成，更新进度
            callback(1., "Done!")
            # 更新文档的 chunk 数量
            DocumentService.increment_chunk_num(
                r["doc_id"], r["kb_id"], tk_count, chunk_count, 0)
            # 记录任务完成信息
            cron_logger.info(
                "Chunk doc({}), token({}), chunks({}), elapsed:{:.2f}".format(
                    r["id"], tk_count, len(cks), timer() - st))


def report_status():
    global CONSUMER_NAME
    while True:
        try:
            obj = REDIS_CONN.get("TASKEXE")
            if not obj: obj = {}
            else: obj = json.loads(obj)
            if CONSUMER_NAME not in obj: obj[CONSUMER_NAME] = []
            obj[CONSUMER_NAME].append(timer())
            obj[CONSUMER_NAME] = obj[CONSUMER_NAME][-60:]
            REDIS_CONN.set_obj("TASKEXE", obj, 60*2)
        except Exception as e:
            print("[Exception]:", str(e))
        time.sleep(30)


if __name__ == "__main__":
    peewee_logger = logging.getLogger('peewee')
    peewee_logger.propagate = False
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)

    exe = ThreadPoolExecutor(max_workers=1)
    exe.submit(report_status)

    while True:
        main()
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
