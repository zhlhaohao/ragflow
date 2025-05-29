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
import json
import re
import traceback
from copy import deepcopy

import trio
from flask import Response, request
from flask_login import current_user, login_required

from api import settings
from api.db import LLMType
from api.db.db_models import APIToken
from api.db.services.conversation_service import ConversationService, structure_answer
from api.db.services.user_service import UserTenantService
from flask import request, Response
from flask_login import login_required, current_user

from api.db import LLMType
from api.db.services.dialog_service import DialogService, chat, chat_nokb, ask, label_question
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle, TenantService
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import get_data_error_result, get_json_result, server_error_response, validate_request
from graphrag.general.mind_map_extractor import MindMapExtractor
from api.utils import ic 
from rag.app.tag import label_question
from mcps.client import mcp_chat


@manager.route("/set", methods=["POST"])  # noqa: F821
@login_required
def set_conversation():
    req = request.json
    conv_id = req.get("conversation_id")
    is_new = req.get("is_new")
    name = req.get("name", "New conversation")

    if len(name) > 255:
        name = name[0:255]

    del req["is_new"]
    if not is_new:
        del req["conversation_id"]
        try:
            if not ConversationService.update_by_id(conv_id, req):
                return get_data_error_result(message="Conversation not found!")
            e, conv = ConversationService.get_by_id(conv_id)
            if not e:
                return get_data_error_result(message="Fail to update a conversation!")
            conv = conv.to_dict()
            return get_json_result(data=conv)
        except Exception as e:
            return server_error_response(e)

    try:
        e, dia = DialogService.get_by_id(req["dialog_id"])
        if not e:
            return get_data_error_result(message="Dialog not found")
        conv = {"id": conv_id, "dialog_id": req["dialog_id"], "name": name, "message": [{"role": "assistant", "content": dia.prompt_config["prologue"]}]}
        ConversationService.save(**conv)
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/get", methods=["GET"])  # noqa: F821
@login_required
def get():
    conv_id = request.args["conversation_id"]
    try:
        e, conv = ConversationService.get_by_id(conv_id)
        if not e:
            return get_data_error_result(message="Conversation not found!")
        tenants = UserTenantService.query(user_id=current_user.id)
        avatar = None
        for tenant in tenants:
            dialog = DialogService.query(tenant_id=tenant.tenant_id, id=conv.dialog_id)
            if dialog and len(dialog) > 0:
                avatar = dialog[0].icon
                break
        else:
            return get_json_result(data=False, message="Only owner of conversation authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)

        def get_value(d, k1, k2):
            return d.get(k1, d.get(k2))

        for ref in conv.reference:
            if isinstance(ref, list):
                continue
            ref["chunks"] = [
                {
                    "id": get_value(ck, "chunk_id", "id"),
                    "content": get_value(ck, "content", "content_with_weight"),
                    "document_id": get_value(ck, "doc_id", "document_id"),
                    "document_name": get_value(ck, "docnm_kwd", "document_name"),
                    "dataset_id": get_value(ck, "kb_id", "dataset_id"),
                    "image_id": get_value(ck, "image_id", "img_id"),
                    "positions": get_value(ck, "positions", "position_int"),
                    "doc_type": get_value(ck, "doc_type", "doc_type_kwd"),
                }
                for ck in ref.get("chunks", [])
            ]

        conv = conv.to_dict()
        conv["avatar"] = avatar
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/getsse/<dialog_id>", methods=["GET"])  # type: ignore # noqa: F821
def getsse(dialog_id):
    token = request.headers.get("Authorization").split()
    if len(token) != 2:
        return get_data_error_result(message='Authorization is not valid!"')
    token = token[1]
    objs = APIToken.query(beta=token)
    if not objs:
        return get_data_error_result(message='Authentication error: API key is invalid!"')
    try:
        e, conv = DialogService.get_by_id(dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")
        conv = conv.to_dict()
        conv["avatar"] = conv["icon"]
        del conv["icon"]
        return get_json_result(data=conv)
    except Exception as e:
        return server_error_response(e)


@manager.route("/rm", methods=["POST"])  # noqa: F821
@login_required
def rm():
    conv_ids = request.json["conversation_ids"]
    try:
        for cid in conv_ids:
            exist, conv = ConversationService.get_by_id(cid)
            if not exist:
                return get_data_error_result(message="Conversation not found!")
            tenants = UserTenantService.query(user_id=current_user.id)
            for tenant in tenants:
                if DialogService.query(tenant_id=tenant.tenant_id, id=conv.dialog_id):
                    break
            else:
                return get_json_result(data=False, message="Only owner of conversation authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)
            ConversationService.delete_by_id(cid)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/list", methods=["GET"])  # noqa: F821
@login_required
def list_convsersation():
    dialog_id = request.args["dialog_id"]
    try:
        if not DialogService.query(tenant_id=current_user.id, id=dialog_id):
            return get_json_result(data=False, message="Only owner of dialog authorized for this operation.", code=settings.RetCode.OPERATING_ERROR)
        convs = ConversationService.query(dialog_id=dialog_id, order_by=ConversationService.model.create_time, reverse=True)

        convs = [d.to_dict() for d in convs]
        return get_json_result(data=convs)
    except Exception as e:
        return server_error_response(e)


@manager.route("/completion", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id", "messages")
def completion():
    req = request.json
    msg = []
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant" and not msg:
            continue

        # 只取用户的消息
        msg.append(m)
    # 获取最后一条历史消息的id
    message_id = msg[-1].get("id")
    try:
        # 获取聊天对象
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(message="Conversation not found!")

        # 深拷贝请求中的messages到会话对象中
        conv.message = deepcopy(req["messages"])
        # 获取助理对象
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")
        del req["conversation_id"]
        del req["messages"]

        if not conv.reference:
            conv.reference = []
        else:

            def get_value(d, k1, k2):
                return d.get(k1, d.get(k2))

            for ref in conv.reference:
                if isinstance(ref, list):
                    continue
                ref["chunks"] = [
                    {
                        "id": get_value(ck, "chunk_id", "id"),
                        "content": get_value(ck, "content", "content_with_weight"),
                        "document_id": get_value(ck, "doc_id", "document_id"),
                        "document_name": get_value(ck, "docnm_kwd", "document_name"),
                        "dataset_id": get_value(ck, "kb_id", "dataset_id"),
                        "image_id": get_value(ck, "image_id", "img_id"),
                        "positions": get_value(ck, "positions", "position_int"),
                        "doc_type": get_value(ck, "doc_type_kwd", "doc_type_kwd"),
                    }
                    for ck in ref.get("chunks", [])
                ]

        if not conv.reference:
            conv.reference = []
        conv.reference.append({"chunks": [], "doc_aggs": []})

        def stream():
            nonlocal dia, msg, req, conv
            yield(" ")
            try:
                # 调用chat函数生成答案，stream模式为True
                for ans in chat(dia, msg, True, **req):
                    # 结构化答案，将答案组装到聊天对象中
                    ans = structure_answer(conv, ans, message_id, conv.id)
                    yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                ConversationService.update_by_id(conv.id, conv.to_dict())
            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        if req.get("stream", True):
            # 生成SSE响应
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        else:
            """ F8080: 非流式响应改造成返回提示词和上下文，用于前端发起请求
            answer = None
            for ans in chat(dia, msg, **req):
                answer = structure_answer(conv, ans, message_id, req["conversation_id"])
                ConversationService.update_by_id(conv.id, conv.to_dict())
                break
            """
            for ans in chat(dia, msg, **req):
                ans = structure_answer(conv, ans, message_id, conv.id)
                return get_json_result(data=ans)
    except Exception as e:
        return server_error_response(e)


@manager.route("/tts", methods=["POST"])  # noqa: F821
@login_required
def tts():
    req = request.json
    text = req["text"]

    tenants = TenantService.get_info_by(current_user.id)
    if not tenants:
        return get_data_error_result(message="Tenant not found!")

    tts_id = tenants[0]["tts_id"]
    if not tts_id:
        return get_data_error_result(message="No default TTS model is set")

    tts_mdl = LLMBundle(tenants[0]["tenant_id"], LLMType.TTS, tts_id)

    def stream_audio():
        try:
            for txt in re.split(r"[，。/《》？；：！\n\r:;]+", text):
                for chunk in tts_mdl.tts(txt):
                    yield chunk
        except Exception as e:
            yield ("data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e)}}, ensure_ascii=False)).encode("utf-8")

    resp = Response(stream_audio(), mimetype="audio/mpeg")
    resp.headers.add_header("Cache-Control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")

    return resp


@manager.route("/delete_msg", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id", "message_id")
def delete_msg():
    req = request.json
    e, conv = ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(message="Conversation not found!")

    conv = conv.to_dict()
    for i, msg in enumerate(conv["message"]):
        if req["message_id"] != msg.get("id", ""):
            continue
        assert conv["message"][i + 1]["id"] == req["message_id"]
        conv["message"].pop(i)
        conv["message"].pop(i)
        conv["reference"].pop(max(0, i // 2 - 1))
        break

    ConversationService.update_by_id(conv["id"], conv)
    return get_json_result(data=conv)


@manager.route("/thumbup", methods=["POST"])  # noqa: F821
@login_required
@validate_request("conversation_id", "message_id")
def thumbup():
    req = request.json
    e, conv = ConversationService.get_by_id(req["conversation_id"])
    if not e:
        return get_data_error_result(message="Conversation not found!")
    up_down = req.get("thumbup")
    feedback = req.get("feedback", "")
    conv = conv.to_dict()
    for i, msg in enumerate(conv["message"]):
        if req["message_id"] == msg.get("id", "") and msg.get("role", "") == "assistant":
            if up_down:
                msg["thumbup"] = True
                if "feedback" in msg:
                    del msg["feedback"]
            else:
                msg["thumbup"] = False
                if feedback:
                    msg["feedback"] = feedback
            break

    ConversationService.update_by_id(conv["id"], conv)
    return get_json_result(data=conv)


@manager.route("/ask", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def ask_about():
    req = request.json
    uid = current_user.id

    def stream():
        nonlocal req, uid
        try:
            for ans in ask(req["question"], req["kb_ids"], uid):
                yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"
        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp


@manager.route("/mindmap", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question", "kb_ids")
def mindmap():
    req = request.json
    kb_ids = req["kb_ids"]
    e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
    if not e:
        return get_data_error_result(message="Knowledgebase not found!")

    embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING, llm_name=kb.embd_id)
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    question = req["question"]
    ranks = settings.retrievaler.retrieval(question, embd_mdl, kb.tenant_id, kb_ids, 1, 12, 0.3, 0.3, aggs=False, rank_feature=label_question(question, [kb]))
    mindmap = MindMapExtractor(chat_mdl)
    mind_map = trio.run(mindmap, [c["content_with_weight"] for c in ranks["chunks"]])
    mind_map = mind_map.output
    if "error" in mind_map:
        return server_error_response(Exception(mind_map["error"]))
    return get_json_result(data=mind_map)


@manager.route("/related_questions", methods=["POST"])  # noqa: F821
@login_required
@validate_request("question")
def related_questions():
    req = request.json
    question = req["question"]
    chat_mdl = LLMBundle(current_user.id, LLMType.CHAT)
    prompt = """
Role: You are an AI language model assistant tasked with generating 5-10 related questions based on a user’s original query. These questions should help expand the search query scope and improve search relevance.

Instructions:
	Input: You are provided with a user’s question.
	Output: Generate 5-10 alternative questions that are related to the original user question. These alternatives should help retrieve a broader range of relevant documents from a vector database.
	Context: Focus on rephrasing the original question in different ways, making sure the alternative questions are diverse but still connected to the topic of the original query. Do not create overly obscure, irrelevant, or unrelated questions.
	Fallback: If you cannot generate any relevant alternatives, do not return any questions.
	Guidance:
	1. Each alternative should be unique but still relevant to the original query.
	2. Keep the phrasing clear, concise, and easy to understand.
	3. Avoid overly technical jargon or specialized terms unless directly relevant.
	4. Ensure that each question contributes towards improving search results by broadening the search angle, not narrowing it.

Example:
Original Question: What are the benefits of electric vehicles?

Alternative Questions:
	1. How do electric vehicles impact the environment?
	2. What are the advantages of owning an electric car?
	3. What is the cost-effectiveness of electric vehicles?
	4. How do electric vehicles compare to traditional cars in terms of fuel efficiency?
	5. What are the environmental benefits of switching to electric cars?
	6. How do electric vehicles help reduce carbon emissions?
	7. Why are electric vehicles becoming more popular?
	8. What are the long-term savings of using electric vehicles?
	9. How do electric vehicles contribute to sustainability?
	10. What are the key benefits of electric vehicles for consumers?

Reason:
	Rephrasing the original query into multiple alternative questions helps the user explore different aspects of their search topic, improving the quality of search results.
	These questions guide the search engine to provide a more comprehensive set of relevant documents.
"""
    ans = chat_mdl.chat(
        prompt,
        [
            {
                "role": "user",
                "content": f"""
Keywords: {question}
Related search terms:
    """,
            }
        ],
        {"temperature": 0.9},
    )
    return get_json_result(data=[re.sub(r"^[0-9]\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]\. ", a)])

@manager.route('/add_messages', methods=['POST'])  # type: ignore # noqa: F821
@login_required
@validate_request("conversation_id", "messages")
def add_messages():
    """ F8080 往对话添加消息-而不是真正进行对话
    """
    req = request.json
    try:
        # 获取聊天对象
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(message="Conversation not found!")

        for m in req["messages"]:
            conv.message.append(m)

        ConversationService.update_by_id(conv.id, conv.to_dict())
        return get_json_result(data=True)

    except Exception as e:
        return server_error_response(e)


@manager.route('/completion_nokb', methods=['POST'])  # type: ignore # noqa: F821
@login_required
@validate_request("conversation_id", "messages")
def completion_nokb():
    """F8080 非知识库对话，例如编程助手对话
    """
    req = request.json
    messages = req["messages"]
    message_id = messages[-1].get("id")

    try:
        # 获取聊天对象
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(message="Conversation not found!")

        # 获取助理对象
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")

        def stream():
            nonlocal dia, messages, conv
            try:
                yield(" ")
                # 调用chat函数生成答案，stream模式为True
                final_ans = None
                for ans in chat_nokb(dia, messages, True):
                    ans["id"] = message_id
                    ans["session_id"] = conv.id
                    final_ans = ans
                    yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"

                # parsed_response = json.loads(final_ans['answer'])
                # if "assistant_reply" not in parsed_response:
                #     parsed_response["assistant_reply"] = ""
                # answer = json.dumps(parsed_response)

                # 将最后一条用户提问和助理的回答保存到对话记录中
                if final_ans is None:
                    err_msg = "大模型返回空回答"
                    yield "data:" + json.dumps({"code": 500, "message": err_msg,
                                            "data": {"answer": "**ERROR**: " + err_msg, "reference": []}},
                                           ensure_ascii=False) + "\n\n"
                else:
                    conv.message.append(messages[-1])
                    conv.message.append({"role": "assistant", "content":
                        final_ans['answer'], "id": message_id})
                    ConversationService.update_by_id(conv.id, conv.to_dict())
            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({"code": 500, "message": str(e),
                                            "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"

            # 全部回答完成
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        if req.get("stream", True):
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        else:
            answer = None
            for ans in chat_nokb(dia, messages, False):
                ConversationService.update_by_id(conv.id, conv.to_dict())
                break

            return get_json_result(data=answer)
    except Exception as e:
        return server_error_response(e)



@manager.route('/list_lite', methods=['GET'])  # type: ignore # noqa: F821
@login_required
def list_convsersation_lite():
    """F8080 为节省流量，不返回对话内容，只返回对话列表

    Returns:
        _type_: _description_
    """
    dialog_id = request.args["dialog_id"]
    try:
        if not DialogService.query(tenant_id=current_user.id, id=dialog_id):
            return get_json_result(
                data=False, message='Only owner of dialog authorized for this operation.',
                code=settings.RetCode.OPERATING_ERROR)
        convs = ConversationService.query(
            dialog_id=dialog_id,
            order_by=ConversationService.model.create_time,
            reverse=True)

        convs = [{k: v for k, v in d.to_dict().items() if k != 'message' and k != 'reference'} for d in convs]
        return get_json_result(data=convs)
    except Exception as e:
        return server_error_response(e)


@manager.route('/completion_mcp', methods=['POST'])  # type: ignore # noqa: F821
@login_required
@validate_request("conversation_id", "messages")
def completion_mcp():
    """F8080 MCP能力加强的对话
    """
    req = request.json
    messages = req["messages"]
    message_id = messages[-1].get("id")

    try:
        # 获取聊天对象
        e, conv = ConversationService.get_by_id(req["conversation_id"])
        if not e:
            return get_data_error_result(message="Conversation not found!")

        # 获取助理对象
        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")

        def stream():
            nonlocal dia, messages, conv
            try:
                yield(" ")
                # 调用chat函数生成答案，stream模式为True
                final_ans = None
                for ans in mcp_chat.MCP_CHAT.chat(dia, messages):
                    ans["id"] = message_id
                    ans["session_id"] = conv.id
                    final_ans = ans
                    yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"

                # 将最后一条用户提问和助理的回答保存到对话记录中
                if final_ans is None:
                    err_msg = "大模型返回空回答"
                    yield "data:" + json.dumps({"code": 500, "message": err_msg,
                                            "data": {"answer": "**ERROR**: " + err_msg, "reference": []}},
                                           ensure_ascii=False) + "\n\n"
                else:
                    conv.message.append(messages[-1])
                    conv.message.append({"role": "assistant", "content":
                        final_ans['answer'], "id": message_id})
                    ConversationService.update_by_id(conv.id, conv.to_dict())
            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({"code": 500, "message": str(e),
                                            "data": {"answer": "**ERROR**: " + str(e), "reference": []}},
                                           ensure_ascii=False) + "\n\n"

            # 全部回答完成
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        if req.get("stream", True):
            resp = Response(stream(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

    except Exception as e:
        return server_error_response(e)
