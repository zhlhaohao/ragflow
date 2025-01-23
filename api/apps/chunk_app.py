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

from flask import request
from flask_login import login_required, current_user

from api.db.services.dialog_service import keyword_extraction, label_question
from rag.app.qa import rmPrefix, beAdoc
from rag.nlp import search, rag_tokenizer
from rag.settings import PAGERANK_FLD
from rag.utils import rmSpace
from api.db import LLMType, ParserType
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import server_error_response, get_data_error_result, validate_request
from api.db.services.document_service import DocumentService
from api import settings
from api.utils.api_utils import get_json_result
import xxhash
import re


@manager.route('/list', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id")
def list_chunk():
    """
    获取文档的分块列表。

    请求参数:
    - doc_id (str): 文档ID。
    - page (int, optional): 分页页码，默认为1。
    - size (int, optional): 每页大小，默认为30。
    - keywords (str, optional): 查询关键词。

    返回:
    - JSON响应，包含分块列表、文档信息和总数量。
    """
    # 获取请求的JSON数据
    req = request.json
    
    # 提取文档ID、页码、每页大小和问题关键词
    doc_id = req["doc_id"]
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req.get("keywords", "")
    
    try:
        # 获取租户ID
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")
    
        # 根据文档ID获取文档
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(message="Document not found!")
    
        # 获取知识库ID列表
        kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
    
        # 构建查询字典
        query = {
            "doc_ids": [doc_id], "page": page, "size": size, "question": question, "sort": True
        }
        if "available_int" in req:
            query["available_int"] = int(req["available_int"])
    
        # 执行搜索
        sres = settings.retrievaler.search(query, search.index_name(tenant_id), kb_ids, highlight=True)
    
        # 初始化结果字典
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
    
        # 遍历搜索结果，构建结果块列表
        for id in sres.ids:
            d = {
                "chunk_id": id,
                "content_with_weight": rmSpace(sres.highlight[id]) if question and id in sres.highlight else sres.field[
                    id].get(
                    "content_with_weight", ""),
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": sres.field[id]["docnm_kwd"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "question_kwd": sres.field[id].get("question_kwd", []),
                "image_id": sres.field[id].get("img_id", ""),
                "available_int": int(sres.field[id].get("available_int", 1)),
                "positions": sres.field[id].get("position_int", []),
            }
            # 确保positions是列表，并且如果存在，每个位置信息都是五元组
            assert isinstance(d["positions"], list)
            assert len(d["positions"]) == 0 or (isinstance(d["positions"][0], list) and len(d["positions"][0]) == 5)
            res["chunks"].append(d)
    
        # 返回搜索结果
        return get_json_result(data=res)
    except Exception as e:
        # 处理异常情况
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, message='No chunk found!',
                                   code=settings.RetCode.DATA_ERROR)
        return server_error_response(e)

@manager.route('/get', methods=['GET'])  # noqa: F821
@login_required
def get():
    """
    获取单个分块的详细信息。

    请求参数:
    - chunk_id (str): 分块ID。

    返回:
    - JSON响应，包含分块的详细信息。
    """
    chunk_id = request.args["chunk_id"]
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")
        tenant_id = tenants[0].tenant_id

        kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
        chunk = settings.docStoreConn.get(chunk_id, search.index_name(tenant_id), kb_ids)
        if chunk is None:
            return server_error_response(Exception("Chunk not found"))
        k = []
        for n in chunk.keys():
            if re.search(r"(_vec$|_sm_|_tks|_ltks)", n):
                k.append(n)
        for n in k:
            del chunk[n]

        return get_json_result(data=chunk)
    except Exception as e:
        if str(e).find("NotFoundError") >= 0:
            return get_json_result(data=False, message='Chunk not found!',
                                   code=settings.RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/set', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id", "chunk_id", "content_with_weight")
def set():
    """
    更新分块的内容。

    请求参数:
    - doc_id (str): 文档ID。
    - chunk_id (str): 分块ID。
    - content_with_weight (str): 分块内容。
    - important_kwd (list, optional): 重要关键词。
    - question_kwd (list, optional): 问题关键词。
    - tag_kwd (list, optional): 标签关键词。
    - tag_feas (list, optional): 标签特征。
    - available_int (int, optional): 可用状态。

    返回:
    - JSON响应，表示操作是否成功。
    """
    req = request.json
    d = {
        "id": req["chunk_id"],
        "content_with_weight": req["content_with_weight"]}
    d["content_ltks"] = rag_tokenizer.tokenize(req["content_with_weight"])
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    if "important_kwd" in req:
        d["important_kwd"] = req["important_kwd"]
        d["important_tks"] = rag_tokenizer.tokenize(" ".join(req["important_kwd"]))
    if "question_kwd" in req:
        d["question_kwd"] = req["question_kwd"]
        d["question_tks"] = rag_tokenizer.tokenize("\n".join(req["question_kwd"]))
    if "tag_kwd" in req:
        d["tag_kwd"] = req["tag_kwd"]
    if "tag_feas" in req:
        d["tag_feas"] = req["tag_feas"]
    if "available_int" in req:
        d["available_int"] = req["available_int"]

    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embd_id)

        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")

        if doc.parser_id == ParserType.QA:
            arr = [
                t for t in re.split(
                    r"[\n\t]",
                    req["content_with_weight"]) if len(t) > 1]
            q, a = rmPrefix(arr[0]), rmPrefix("\n".join(arr[1:]))
            d = beAdoc(d, arr[0], arr[1], not any(
                [rag_tokenizer.is_chinese(t) for t in q + a]))

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"] if not d.get("question_kwd") else "\n".join(d["question_kwd"])])
        v = 0.1 * v[0] + 0.9 * v[1] if doc.parser_id != ParserType.QA else v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        settings.docStoreConn.update({"id": req["chunk_id"]}, d, search.index_name(tenant_id), doc.kb_id)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/switch', methods=['POST'])  # noqa: F821
@login_required
@validate_request("chunk_ids", "available_int", "doc_id")
def switch():
    """
    切换分块的可用状态。

    请求参数:
    - chunk_ids (list): 分块ID列表。
    - available_int (int): 可用状态。
    - doc_id (str): 文档ID。

    返回:
    - JSON响应，表示操作是否成功。
    """
    req = request.json
    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        for cid in req["chunk_ids"]:
            if not settings.docStoreConn.update({"id": cid},
                                                {"available_int": int(req["available_int"])},
                                                search.index_name(DocumentService.get_tenant_id(req["doc_id"])),
                                                doc.kb_id):
                return get_data_error_result(message="Index updating failure")
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])  # noqa: F821
@login_required
@validate_request("chunk_ids", "doc_id")
def rm():
    """
    删除分块。

    请求参数:
    - chunk_ids (list): 分块ID列表。
    - doc_id (str): 文档ID。

    返回:
    - JSON响应，表示操作是否成功。
    """
    req = request.json
    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        if not settings.docStoreConn.delete({"id": req["chunk_ids"]}, search.index_name(current_user.id), doc.kb_id):
            return get_data_error_result(message="Index updating failure")
        deleted_chunk_ids = req["chunk_ids"]
        chunk_number = len(deleted_chunk_ids)
        DocumentService.decrement_chunk_num(doc.id, doc.kb_id, 1, chunk_number, 0)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/create', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id", "content_with_weight")
def create():
    """
    创建新的分块。

    请求参数:
    - doc_id (str): 文档ID。
    - content_with_weight (str): 分块内容。
    - important_kwd (list, optional): 重要关键词。
    - question_kwd (list, optional): 问题关键词。

    返回:
    - JSON响应，包含新创建的分块ID。
    """
    req = request.json
    chunck_id = xxhash.xxh64((req["content_with_weight"] + req["doc_id"]).encode("utf-8")).hexdigest()
    d = {"id": chunck_id, "content_ltks": rag_tokenizer.tokenize(req["content_with_weight"]),
         "content_with_weight": req["content_with_weight"]}
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["important_kwd"] = req.get("important_kwd", [])
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(req.get("important_kwd", [])))
    d["question_kwd"] = req.get("question_kwd", [])
    d["question_tks"] = rag_tokenizer.tokenize("\n".join(req.get("question_kwd", [])))
    d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.datetime.now().timestamp()

    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        d["kb_id"] = [doc.kb_id]
        d["docnm_kwd"] = doc.name
        d["title_tks"] = rag_tokenizer.tokenize(doc.name)
        d["doc_id"] = doc.id

        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")

        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        if not e:
            return get_data_error_result(message="Knowledgebase not found!")
        if kb.pagerank:
            d[PAGERANK_FLD] = kb.pagerank

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING.value, embd_id)

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"] if not d["question_kwd"] else "\n".join(d["question_kwd"])])
        v = 0.1 * v[0] + 0.9 * v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        settings.docStoreConn.insert([d], search.index_name(tenant_id), doc.kb_id)

        DocumentService.increment_chunk_num(
            doc.id, doc.kb_id, c, 1, 0)
        return get_json_result(data={"chunk_id": chunck_id})
    except Exception as e:
        return server_error_response(e)


@manager.route('/retrieval_test', methods=['POST'])  # noqa: F821
@login_required
@validate_request("kb_id", "question")
def retrieval_test():
    """
    进行检索测试。

    请求参数:
    - kb_id (list): 知识库ID列表。
    - question (str): 查询问题。
    - doc_ids (list, optional): 文档ID列表。
    - similarity_threshold (float, optional): 相似度阈值，默认为0.0。
    - vector_similarity_weight (float, optional): 向量相似度权重，默认为0.3。
    - top_k (int, optional): 顶部结果数量，默认为1024。
    - rerank_id (str, optional): 重排序模型ID。
    - keyword (bool, optional): 是否使用关键词提取，默认为False。
    - highlight (bool, optional): 是否高亮显示，默认为False。

    返回:
    - JSON响应，包含检索结果。
    """
    req = request.json
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req["question"]
    kb_ids = req["kb_id"]
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    doc_ids = req.get("doc_ids", [])
    similarity_threshold = float(req.get("similarity_threshold", 0.0))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    top = int(req.get("top_k", 1024))
    tenant_ids = []

    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        for kb_id in kb_ids:
            for tenant in tenants:
                if KnowledgebaseService.query(
                        tenant_id=tenant.tenant_id, id=kb_id):
                    tenant_ids.append(tenant.tenant_id)
                    break
            else:
                return get_json_result(
                    data=False, message='Only owner of knowledgebase authorized for this operation.',
                    code=settings.RetCode.OPERATING_ERROR)

        e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
        if not e:
            return get_data_error_result(message="Knowledgebase not found!")

        embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)

        rerank_mdl = None
        if req.get("rerank_id"):
            rerank_mdl = LLMBundle(kb.tenant_id, LLMType.RERANK.value, llm_name=req["rerank_id"])

        if req.get("keyword", False):
            chat_mdl = LLMBundle(kb.tenant_id, LLMType.CHAT)
            question += keyword_extraction(chat_mdl, question)

        labels = label_question(question, [kb])
        retr = settings.retrievaler if kb.parser_id != ParserType.KG else settings.kg_retrievaler
        ranks = retr.retrieval(question, embd_mdl, tenant_ids, kb_ids, page, size,
                               similarity_threshold, vector_similarity_weight, top,
                               doc_ids, rerank_mdl=rerank_mdl, highlight=req.get("highlight"),
                               rank_feature=labels
                               )
        for c in ranks["chunks"]:
            c.pop("vector", None)
        ranks["labels"] = labels

        return get_json_result(data=ranks)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, message='No chunk found! Check the chunk status please!',
                                   code=settings.RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/knowledge_graph', methods=['GET'])  # noqa: F821
@login_required
def knowledge_graph():
    """
    获取知识图谱信息。

    请求参数:
    - doc_id (str): 文档ID。

    返回:
    - JSON响应，包含知识图谱和思维导图信息。
    """
    doc_id = request.args["doc_id"]
    tenant_id = DocumentService.get_tenant_id(doc_id)
    kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
    req = {
        "doc_ids": [doc_id],
        "knowledge_graph_kwd": ["graph", "mind_map"]
    }
    sres = settings.retrievaler.search(req, search.index_name(tenant_id), kb_ids)
    obj = {"graph": {}, "mind_map": {}}
    for id in sres.ids[:2]:
        ty = sres.field[id]["knowledge_graph_kwd"]
        try:
            content_json = json.loads(sres.field[id]["content_with_weight"])
        except Exception:
            continue

        if ty == 'mind_map':
            node_dict = {}

            def repeat_deal(content_json, node_dict):
                if 'id' in content_json:
                    if content_json['id'] in node_dict:
                        node_name = content_json['id']
                        content_json['id'] += f"({node_dict[content_json['id']]})"
                        node_dict[node_name] += 1
                    else:
                        node_dict[content_json['id']] = 1
                if 'children' in content_json and content_json['children']:
                    for item in content_json['children']:
                        repeat_deal(item, node_dict)

            repeat_deal(content_json, node_dict)

        obj[ty] = content_json

    return get_json_result(data=obj)
