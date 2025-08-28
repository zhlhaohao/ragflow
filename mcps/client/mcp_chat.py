from api.utils.log_utils import initRootLogger
initRootLogger("ragflow_server")

import asyncio
import threading
import json
import logging
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from api import settings
from mcps.client.lite_llm_json import LiteLLMJson
from api.db.services.llm_service import TenantLLMService, LLMBundle
from api.db import LLMType
import mcp
from fastmcp import Client
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams
import re
import queue
import toml
from api.utils import ic
import datetime
from copy import deepcopy
import time
from api.db.services.dialog_service import retrieval
from api.utils import colored_log_message

MCP_CHAT = None

async def init_mcp() -> None:
    global MCP_CHAT
    if  MCP_CHAT is None:
        MCP_CHAT = McpChat()

    return await MCP_CHAT.init_servers()

def get_current_time_with_weekday() -> str:
    """
    获取当前时间并格式化为 yyyy-mm-dd hh:mm:ss 加上星期几

    Returns:
        str: 格式化后的时间字符串，例如 "2023-10-05 14:30:45 Thursday"
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %A")
    return formatted_time

json_schema = {
    "type": "object",
    "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}},
    "required": ["tool"],
}
llm_json = LiteLLMJson(json_schema)

class ToolRequest(BaseModel):
    tool: str
    arguments: dict


class Configuration:
    """
    Manages configuration and environment variables for the MCP client.
    """

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.
        """
        with open(file_path, "r") as f:
            return toml.load(f)


class Server:
    """
    管理单个 MCP server connections and tool execution.
    """
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.tools: list[Any] = []

    def get_client(self, sampling_handler = None):
        if "command" in self.config:
            config = {"mcpServers": {self.name: {
                        "command": self.config["command"],
                        "args": self.config["args"],
                    }
                }
            }
        if "url" in self.config:
            config = {
                "mcpServers": {
                    self.name: {
                        "url": self.config["url"],
                        "transport": self.config["transport"],
                    }
                }
            }

        if sampling_handler:
            client = Client(config, sampling_handler = sampling_handler)
        else:
            client = Client(config)

        return client


    def judge_permission(self, tool_name: str, user_email: str) -> bool:
        """
        检查用户是否有权限使用指定的工具。

        参数:
            tool_name (str): 工具名称。
            user_email (str): 用户的电子邮件地址。

        返回:
            bool: 如果用户有权限，返回 True；否则返回 False。
        """
        # 获取 permission 字段，如果不存在则默认为空字典
        permissions = self.config.get("permission", {})

        # 检查工具名是否在 permissions 中，并且用户邮箱是否在对应的邮箱列表中
        if tool_name in permissions and user_email not in permissions[tool_name]:
            return False

        return True

    async def list_tools(self) -> list[Any]:
        tools = []
        client = self.get_client()
        async with client:
            if client.is_connected():
                logging.info(colored_log_message(f"103- mcp server {self.name} 连接{client.is_connected()}","green"))
            resp = await client.list_tools()
            for tool in resp:
                tools.append(Tool(tool.name, tool.description, tool.inputSchema))
        self.tools = tools
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        dialog = None,
        history = None,
        chat_mdl = None,
        msg_queue = None,
        retries: int = 2,
        delay: float = 2,
    ) -> Any:
        """带重试机制的调用工具.
        """
        # 接收到工具中间结果输出
        async def sampling_handler(
            messages: list[SamplingMessage],
            params: SamplingParams,
            ctx: RequestContext,
        ):
            content = messages[0].content.text
            try:
                result = json.loads(content)
            except Exception:
                result = content

            # 如果是字符串，可能是/开头的指令或者是logger message
            if isinstance(result, str):
                if str.startswith(result, "/history"):
                    if history:
                        return json.dumps({"history": history},ensure_ascii=False)
                    else:
                        return "history not found"
                else:
                    if msg_queue and result:
                        if result.startswith("/log:"):
                            result = result[5:]
                        msg_queue.put(result)
                    return ""

            # 如果是协助mcp server调用llm
            system_prompt = "you are a helpful assistant."
            if result.get('keep_system'):
                system_prompt = dialog.prompt_config['system']

            messages = []

            if result.get('system'):
                messages.extend(result.get('system'))

            # 如果需要保留历史对话
            if result.get('keep_history'):
                # 把所有的历史消息（包括系统消息）添加到消息队列中
                if result.get('keep_system'):
                    messages.extend(history)
                else:
                    # 剔除掉系统消息，把用户、助理的历史消息放到此次消息队列中
                    filtered_history = [msg for msg in history if msg["role"] != "system"]
                    messages.extend(filtered_history)

            # 删除历史消息中的最后一条用户消息
            if not result.get('keep_last_message'):
                last_message = messages[-1]
                if last_message["role"] == "user":
                    messages.pop()

            # 粘贴mcp server过来的用户、助理消息
            if result.get('messages'):
                messages.extend(result.get('messages'))

            gen_conf = deepcopy(dialog.llm_setting)
            if result.get('enable_json'):
                if "deepseek-r1-250528" in chat_mdl.llm_name.lower() or "deepseek-reasoner" in chat_mdl.llm_name.lower():
                    gen_conf['response_format'] = {
                        'type': 'json_object'
                    }

            for ans in chat_mdl.chat_streamly(system_prompt, messages, gen_conf):
                if msg_queue and len(ans)>0:
                    msg_queue.put(ans)

            logging.info(f"166- response_content: {ans}")
            return ans

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"136- Executing {tool_name}...")
                client = self.get_client(sampling_handler)

                # logger.info(f"89- 调用工具:{tool_name}, 参数是:{tool_args}\n")
                async with client:
                    """
                       list[
                            mcp.types.TextContent | mcp.types.ImageContent | mcp.types.EmbeddedResource
                        ]
                    """
                    resp = await client.call_tool(tool_name, arguments)
                    result = resp[0]
                    if isinstance(result, mcp.types.TextContent):
                        data = result.text
                    if isinstance(result, mcp.types.ImageContent):
                        data = f"""![](data:image/{result.mimeType};base64,{result.data})"""

                    # logging.info(f"101- 工具返回结果:\n{data[0:100]}")
                    return data

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"159- Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    # logging.info(f"162- Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("165- Max retries reached. Failing.")
                    raise


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
### Tool: {self.name}
#### Description: {self.description}
#### Arguments:
{chr(10).join(args_desc)}
"""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
        }

class McpChat:
    server_config = None    # 所有mcp server的配置
    server_tools = {}

    async def init_servers(self):
        """
        管理多个mcp server
        """
        if self.server_config is None:
            config = Configuration()
            self.server_config = config.load_config("conf/mcp_config.toml")

            self.servers = [
                Server(name, srv_config)
                for name, srv_config in self.server_config["mcpServers"].items()
            ]

        is_success = True
        for server in self.servers:
            # 如果没有列出出工具，则尝试列出
            if not self.server_tools.get(server.name, False):
                try:
                    tools = await server.list_tools()
                    self.server_tools[server.name] = tools
                    server.config["tools"] = [tool.to_dict() for tool in tools]
                    # logging.error(f"Success loading tools for mcp server {server.name}")
                except Exception as e:
                    is_success = False
                    # logging.error(f"Error loading tools for mcp server {server.name}: {e}")
                    continue

        return is_success

    def get_server(self, server_name):
        for server in self.servers:
            if server.name == server_name:
                return server
        return None

    def get_visible_servers(self, email):
        result = {}
        for server in self.servers:
            if "visible" not in server.config or email in server.config['visible']:
                result[server.name] = server.config
        return result

    def mcp_instruction(self, mcp_servers, user_email):
        """组装当前用户的当前可用mcp_servers的工具提示，检查了工具的权限"""
        tools_description = ''
        for server_name in mcp_servers:
            tools_description += f"\n\n## Tools of mcp server {server_name}:"
            tools = self.server_tools.get(server_name,[])
            server = self.get_server(server_name)
            temp_list = []
            for tool in tools:
                if server.judge_permission(tool.name, user_email):
                    temp_list.append(tool.format_for_llm())
            tools_description += "\n".join(temp_list)

            config = self.server_config["mcpServers"].get(server_name,{})
            system_prompt = config.get("system")
            if system_prompt:
                tools_description += f"\n### Suggestion or extra information of mcp server {server_name}:\n{system_prompt}"

        # today_desc = get_current_time_with_weekday()
        instruction = f"""You are a helpful assistant with access to these tools before answering user's question:

{tools_description}

## Tool call guideline:
- Choose the appropriate tool based on the user's question.
- At each step only one tool is called, multiple tools are called in multiple steps.
- ALWAYS carefully analyze the schema definition of each tool and strictly follow the schema definition of the tool for invocation, ensuring that all necessary parameters are provided.
- If the tool fails, check whether there is any error in the tool call according to the error returned, for example, if there is any error in the tool name or the arguments, and retry the tool call in the correct way. If you judge that this is not your problem but a system problem, such as a network connection error, return directly, do not call tool again.
- OUTPUT FORMAT: You must ONLY Respond strictly in **JSON** and nothing else.The response should adhere to the following JSON schema:

{{
"tool": "string",
"arguments": "dict"
}}


## Final answer:
- Answering User questions should include Thought regardless of whether or not you need to call a tool.
- ALWAYS start with a Thought and Only ONE Thought at a time.
- You should keep repeating the above steps till you have enough information to answer the question without using any more tools. At That Moment, YOU MUST respond with plain text: <FINAL_ANSWER>  -- Do not write any other words

## Extra instructions:
- Transform the tool data into a natural, conversational response, avoid simply repeating the tool data
- Keep responses concise but informative
- Focus on the most relevant information
- If tool data is table data and the user does not specify a visualize type, you should always transform data into a markdown table
- If tool data is an image url, you should transform into markdown format: ![image explanation](image url)
- If the tool's response is base64 image, do not repeat the base64 code.
- If user uploaded an image，call read_image tool.
- If user's question mentioned an image，please search the image the user has just uploaded and call read_image tool.
- If user uploaded an pdf file，call parse document tool.
"""
        return instruction


    async def mcp_tool_call(
            self,
            current_user,
            server: Server,
            tool_call,
            dialog = None,
            history = None,
            chat_mdl = None,
            msg_queue = None) -> str:
        """分析llm的回答，如果需要则调用MCP工具.

        Args:
            llm_response: The response from the LLM.

        Returns:
            工具执行结果或者是入参
        """

        logging.info(f"279- Executing tool: {tool_call['tool']}")
        logging.info(f"280- With arguments: {tool_call['arguments']}")
        try:
            if not server.judge_permission(tool_call["tool"], current_user.email):
                raise Exception(f"{current_user.email} does not have permission to use tool {tool_call['tool']}")

            result = await server.execute_tool(
                tool_call["tool"], tool_call["arguments"], dialog, history, chat_mdl, msg_queue
            )
            if "data:image" in result:
                return f"\n\n{result}"
            else:
                result = self.convert_mixed_utf_string(result)
                return result
        except Exception as e:
            error_msg = f"291- 工具执行出错: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def convert_mixed_utf_string(self, input_str):
        """
        处理混杂了 UTF-8 和 UTF 转义字符的字符串，将其正确转换为 UTF-8 字符串

        参数:
        input_str (str): 包含混合编码的输入字符串

        返回:
        str: 转换后的纯 UTF-8 字符串
        """
        try:
            # 使用正则表达式查找所有 \uXXXX 格式的转义序列
            def replace_escape(match):
                # 获取转义序列中的 Unicode 码点
                escape_code = match.group(1)
                # 转换为对应的 Unicode 字符
                return chr(int(escape_code, 16))

            # 替换所有找到的转义序列
            decoded_str = re.sub(r'\\u([0-9a-fA-F]{4})', replace_escape, input_str)

            return decoded_str
        except Exception as e:
            print(f"处理字符串时出错: {e}")
            # 如果处理失败，返回原始字符串或进行其他错误处理
            return input_str


    def chat(self, dialog, messages, current_user):
        """
        Main chat session handler.
        """
        question = messages[-1]["content"]
        # if "**UPLOAD**" in question:
        #     question = f"I have just {question}, call the parse document tool."

        llm_id, model_provider = TenantLLMService.split_model_name_and_factory(dialog.llm_id)
        # 从TenantLLM表取出模型信息（包括api_key）,然后封装成对象返回，也包装了chat_streamly和chat方法
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
        if not chat_mdl:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)

        # mcp_chat_mdl是专用于mcp tool工具解析的模型，不是用于最终问题回答的模型，由于qwen3系列对tool解析较好，所以不用单独新建解析模型，共用对话模型即可
        # if "qwen3" in dialog.llm_id.lower():
        #     mcp_chat_mdl = chat_mdl
        # else:
        mcp_chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, settings.MCP_TOOL_MDL)
        if not mcp_chat_mdl:
            mcp_chat_mdl = chat_mdl

        system_prompt = dialog.prompt_config["system"]

        gen_conf = dialog.llm_setting
        dia_mcp_servers = gen_conf.get("mcp_servers")
        mcp_instruction = self.mcp_instruction(dia_mcp_servers, current_user.email)

        def truncate_messages(messages):
            new_messages = deepcopy(messages)
            for msg in new_messages:
                if ("Tool response" in msg["content"]) and len(msg["content"]) > 1024:
                    msg["content"] = msg["content"][:1024] + "...[truncated]"
            return new_messages

        history_msgs_json = json.dumps(messages[:-1],ensure_ascii=False)
        agent_prompt = f"{mcp_instruction}\n\n## Current Conversation\nBelow is the current conversation consisting of interleaving human and assistant messages. Think step by step.\n\n{history_msgs_json}\n"
        mcp_messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nQuery:\n{question}"
            }
        ]

        mcp_ans = ""
        last_tool_call = ""
        msg_queue = queue.Queue()
        result_container = [None]
        while True:
            try:
                mcp_gen_conf = {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    # "top_k": 5,
                    "enable_cot": False,
                }

                msgs = truncate_messages(mcp_messages)
                logging.info(f"529- {mcp_chat_mdl.llm_name} input:\n{msgs}")
                response_content = mcp_chat_mdl.chat(agent_prompt, msgs, mcp_gen_conf)


            except Exception as e:
                logging.error(f"406- **ERROR** {str(e)}")

            logging.info(f"536- {mcp_chat_mdl.llm_name}: %s", response_content)
            mcp_server = None
            try:
                # 解析出工具调用对象
                tool_call = llm_json.parse_response(response_content)
                # 如果llm要求使用工具
                if "tool" in tool_call:
                    # 遍历查找工具所对应的mcp server
                    for server in self.servers:
                        if server.name in dia_mcp_servers and any(tool.name == tool_call["tool"] for tool in server.tools):
                            mcp_server = server
                            if "arguments" not in tool_call:
                                tool_call["arguments"] = {}

                            tool_json = json.dumps(tool_call, ensure_ascii=False)

                            if tool_json != last_tool_call:
                                last_tool_call = tool_json

                                logging.info(f"459- 调用MCP工具{server.name}:\n{tool_json}")
                                mcp_ans  += f"调用MCP工具{server.name}:\n\n```json\n{tool_json}\n```\n\n"
                                yield {"answer": mcp_ans}
                            else:
                                # 如果跟上次的工具调用一模一样，则不重复调用
                                mcp_server = None
                                logging.warning("448- 重复调用相同工具，忽略")

                            break
                else:
                    logging.info("463- 无需调用工具")

            except Exception:
                logging.info("466- 无需调用工具")

            # 如果要求使用tool
            if mcp_server is not None:
                # 异步执行mcp工具调用
                async def run_async_func():
                    result = await self.mcp_tool_call(current_user, mcp_server, tool_call, dialog, messages, chat_mdl, msg_queue)
                    result_container[0] = result
                    # 执行结束标记
                    msg_queue.put(None)

                # 新建事件循环（协程）中启动mcp调用
                def start_event_loop():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(run_async_func())


                # 在新的线程中，启动协程，这样就确保了同步函数中没有任何异步代码
                thread = threading.Thread(target=start_event_loop)
                thread.start()

                # 循环获取中间结果队列
                while True:
                    try:
                        msg = msg_queue.get(timeout=0.5)
                        if msg is None:
                            break  # 收到结束信号
                        yield {"answer": f"{mcp_ans}\n{msg}"}
                    except queue.Empty:
                        # 如果线程已经结束，则退出循环
                        if not thread.is_alive():
                            break

                # 等待线程结束
                thread.join()
                response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)

                # 将工具调用命令和结果附加到历史消息数组
                tool_response = result_container[0].replace(r'\\u', r'\u')
                logging.info(f"492-工具执行结果：\n{tool_response[:1024]}")

                # search_knowledgebase需要内部搜索知识库
                if tool_call["tool"] == "search_knowledgebase":
                    tool_response = retrieval(
                        dialog,
                        tool_call["arguments"]["query"],
                    )

                mcp_messages.append(
                    {"role": "assistant", "content": f"Tool call:\n{json.dumps(tool_call, ensure_ascii=False)}" }
                )
                mcp_messages.append(
                    {"role": "assistant", "content": f"Tool response：\n\n{tool_response}" }
                )

                # 如果工具返回了最终答案，则结束会话
                if tool_response.startswith("<FINAL_ANSWER>"):
                    tool_response = tool_response.replace("<FINAL_ANSWER>", "")
                    mcp_ans += f"{tool_response}"
                    yield {"answer": mcp_ans}
                    break   # 退出会话
                else:
                    # 如果工具返回了中间结果，则继续下一个循环,LLM会继续选择合适的工具
                    if len(tool_response)>1024:
                        ignoreLen = len(tool_response)-1024
                        tool_response = tool_response[:1024] + f"\n\n省略{ignoreLen}字..."

                    mcp_ans += f"\n\n工具执行结果:\n\n```\n{tool_response}\n```\n\n"
                    yield {"answer": mcp_ans}

            # 没有使用tool，表示已经收集了足够的信息，可以回答用户问题了
            else:
                # 用正式对话的模型重新问一次
                if ("<FINAL_ANSWER>" in response_content or mcp_chat_mdl != chat_mdl):
                    final_prompt = f"{system_prompt}\n\n## Current Conversation\nBelow is the current conversation consisting of interleaving human and assistant messages.\n\n{history_msgs_json}\n"

                    logging.info(f"666- system prompt:\n{final_prompt}")
                    logging.info(f"667- llm input:\n{mcp_messages}")

                    for ans in chat_mdl.chat_streamly(final_prompt, mcp_messages, gen_conf):
                        yield {"answer": f"{mcp_ans}\n{ans}"}

                    response_content = ans

                if "ERROR" in response_content:
                    response_content += "\n\n**有错误发生，可能是因为上下文长度超限**"

                logging.info(f"639- {chat_mdl.llm_name}最终回答:\n{ans}")
                yield {"answer": f"{mcp_ans}\n{response_content}"}
                break


async def main() -> None:
    """Initialize and run the chat session."""
    logging.info("start mcp chat...")
    settings.init_settings()
    mcp_chat = McpChat()
    await mcp_chat.init_servers()

    await mcp_chat.start()

if __name__ == "__main__":

    asyncio.run(main())
