#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import asyncio
import logging
import re
import aiohttp
import os
import random
from rag.nlp import rag_tokenizer
from api.utils import ic
import copy
from rag.nlp import add_positions, tokenize, tokenize_table, naive_merge
from rag.app.naive import Markdown

MinerU_API_BASE_URL=os.getenv("MinerU_API_BASE_URL", "http://127.0.0.1:8000")

async def mineru_parse(file_path, binary=None):
    file_name = os.path.basename(file_path)
    # random_number = random.randint(100000, 999999)  # Generate 6-digit random number
    # md_file_path = os.path.join(TEMP_DIR, f"{file_name}.{random_number}.md")
    filetype = os.path.splitext(file_path)[1][1:]

    if not binary:
        with open(file_path, "rb") as f:
            binary = f.read()

    form_data = aiohttp.FormData()
    form_data.add_field(
        "file",
        binary,
        filename=file_name,
        content_type=f"application/{filetype}",
    )
    parse_method = "auto"  # ocr
    form_data.add_field("parse_method", parse_method)

    # MinerU api url
    api_url = MinerU_API_BASE_URL + "/file_parse"
    timeout = aiohttp.ClientTimeout(total=3600)

    # 文档解析任务
    async def parse_document_task(session):
        async with session.post(api_url, data=form_data) as response:
            if response.status != 200:
                error_text = await response.text()
                err_msg = f"解析返回错误: {response.status}：{error_text}"
                logging.error(err_msg)
                raise Exception(err_msg)

            result = await response.json()
            if "error" in result:
                raise Exception(result["error"])

            content = result.get("md_content")
            return content

    # 进度汇报任务
    async def report_progress():
        tick_counter = 0
        while True:
            await asyncio.sleep(2)
            tick_counter += 1
            # with suppress(Exception):
            #     await ctx.sample(f"已等待{tick_counter * 2}秒")

    # 开始执行解析和报告任务
    async with aiohttp.ClientSession(timeout=timeout) as session:
        parse_task = asyncio.create_task(parse_document_task(session))
        progress_task = asyncio.create_task(report_progress())

        # 等待主任务的完成
        done, _ = await asyncio.wait(
            [parse_task], return_when=asyncio.FIRST_COMPLETED
        )

        # 取消进度报告任务，避免在解析任务完成后继续执行不必要的等待和报告操作
        progress_task.cancel()
        # 使用suppress上下文管理器忽略asyncio.CancelledError异常， 防止因任务被取消而抛出异常影响主流程执行
        # with suppress(asyncio.CancelledError):
        #     await progress_task

        if parse_task.exception():
            raise parse_task.exception()

        return parse_task.result()


def tokenize_chunks(text_chunks, context_chunks, doc, eng):
    res = []
    for ii, text in enumerate(text_chunks):
        if len(text.strip()) == 0:
            continue
        logging.debug("-- {}".format(text))

        # 将doc metainfo添加到数据中
        data = copy.deepcopy(doc)
        add_positions(data, [[ii]*5])
        # 分词，将分词后的结果添加到数据中
        tokenize(data, text, eng)

        # 这个字段如果存在，那么embedding就采用这个字段
        if context_chunks:
            data["context"] = context_chunks[ii]
        res.append(data)
    return res


def chunk(file_path, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, excel, txt.
        This method apply the naive ways to chunk files.
        Successive text will be sliced into pieces using 'delimiter'.
        Next, these successive pieces are merge into chunks whose token number is no more than 'Max token number'.
    """
    try:
        parser_config = kwargs.get(
            "parser_config", {
                "chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})

        filename = os.path.basename(file_path)
        is_english = lang.lower() == "english"
        doc = {
            "docnm_kwd": filename,
            "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
        }
        doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
        res = []

        if re.search(r"\.pdf$", filename, re.IGNORECASE):
            callback(0.1, "开始解析.")
            content = asyncio.run(mineru_parse(file_path, binary))
            md_filename = filename + ".md"

            sections, tables = Markdown(int(parser_config.get("chunk_token_num", 512)))(md_filename, content)
            res = tokenize_table(tables, doc, is_english)
            text_chunks = naive_merge(
                sections, int(parser_config.get(
                    "chunk_token_num", 512)), parser_config.get(
                    "delimiter", "\n!?。；！？"))
            context_chunks = None

            # for i, ck in enumerate(result.chunks()):
            #     text = ck.to_context_text()
            #     # print(f"Chunk #{i+1} ({ck.start_page}-{ck.end_page}):")

            callback(0.8, "完成解析")

        else:
            raise NotImplementedError(
                "file type not supported yet(pdf, xlsx, doc, docx, txt supported)")

        res = []
        res.extend(tokenize_chunks(text_chunks, context_chunks, doc, is_english))
        return res
    except Exception as e:
        logging.error(f"pdf解析发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass

    filename = "/home/lianghao/github/ragflow/temp/中国联通5G与工业控制协同技术白皮书.pdf"
    # filename = "/home/lianghao/github/LazyLLM/tests/example.md"
    res = chunk(filename, from_page=0, to_page=100, callback=dummy)
    for item in res:
        print(item["content_with_weight"],"\n---------\n")
    pass


"""

res = List()
list item in res:

pdf file:
{
"docnm_kwd": "/home/lianghao/github/LazyLLM/tests/领域微调实践.pdf",
"title_tks": "home lianghao github lazyllm test 领域 微调 实践",
"title_sm_tks": "home lianghao github lazyllm test 领域 微调 实践",
"image": <PIL.Image.Image image mode=RGB size=1270x999 at 0x723A64142F20>,
"page_num_int": [2, 3, 3, 3],
"position_int": [(...), (...), (...), (...)],
"top_int": [739, 75, 124, 157],
"content_with_weight": "模型能力的影响：",
"content_ltks": "模型 能力 的 影响",
"content_sm_ltks": "模型 能力 的 影响"}


md file:
{'docnm_kwd': '/home/lianghao/github/LazyLLM/tests/example.md',
'title_tks': 'home lianghao github lazyllm test exampl',
'title_sm_tks': 'home lianghao github lazyllm test exampl',
'page_num_int': [16],
'position_int': [(...)],
'top_int': [15],
'content_with_weight': '### 实际集成案例
',
'content_ltks': '实际 集成 案例',
'content_sm_ltks': '实际 集成 案例'}

"""
