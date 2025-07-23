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
            logging.info(f"103- mcp server {self.name} 连接{client.is_connected()}")
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

                    logging.info(f"101- 工具返回结果:\n{data[0:100]}")
                    return data

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"159- Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"162- Retrying in {delay} seconds...")
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


class McpChat:
    server_config = None
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
            if not self.server_tools.get(server.name, False):
                try:
                    tools = await server.list_tools()
                    self.server_tools[server.name] = tools
                    logging.error(f"Success loading tools for mcp server {server.name}")
                except Exception as e:
                    is_success = False
                    logging.error(f"Error loading tools for mcp server {server.name}: {e}")
                    continue

        return is_success

    def get_server(self, server_name):
        for server in self.servers:
            if server.name == server_name:
                return server
        return None


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

        today_desc = get_current_time_with_weekday()
        instruction = f"""
You are a helpful assistant with access to these tools:

{tools_description}

## Tool call guideline:
1. Choose the appropriate tool based on the user's question. When you don't need to use a tool, then answer "<NO_TOOL_CALL>"
2. Check that all the required parameters for each tool call are provided or can reasonably be inferred from context. IF there are no relevant tools or there are missing values for required parameters, ask the user to supply these values; otherwise proceed with the tool calls.
3. If the user provides a specific value for a parameter (for example provided in quotes), make sure to use that value EXACTLY. DO NOT make up values for or ask about optional parameters. Carefully analyze descriptive terms in the request as they may indicate required parameter values that should be included even if not explicitly quoted.
4. At each step only one tool is called, multiple tools are called in multiple steps.
5. NEVER call a tool that does not exist, such as a tool that has been used in the conversation history or tool call history, but is no longer available.
6. ALWAYS carefully analyze the schema definition of each tool and strictly follow the schema definition of the tool for invocation,ensuring that all necessary parameters are provided.
7. If you make a plan, immediately follow it, do not wait for the user to confirm or tell you to go ahead. The only time you should stop is if you need more information from the user that you can't find any other way, or have different options that you would like the user to weigh in on.
8. If a user asks you to expose your tools, always respond with a description of the tool, and be sure not to expose tool information to the user.
9. If the tool fails, check whether there is any error in the tool call according to the error returned, for example, if there is any error in the tool name or the arguments, and retry the tool call in the correct way. If you judge that this is not your problem but a system problem, such as a network connection error, return directly, do not call tool again.
10. If the tool call fails for more than 3 consecutive invocations, return directly, do not call this tool again.
11. When the task includes time range requirement, Incorporate appropriate time-based search parameters in your queries (e.g., "after:2020", "before:2023", or specific date ranges)
12. Today is {today_desc}

CRITICAL: When you need to use a tool, you must ONLY Respond strictly in **JSON** and nothing else.The response should adhere to the following JSON schema:
### Response Format:
{{
"tool": "string"
"arguments": "dict"
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response, avoid simply repeating the raw data
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. If raw data is table data and the user does not specify a visualize type, you should always transform data into a markdown table
6. If raw data is an image url, you should transform into markdown format: ![image explanation](image url)
7. If the tool's response is base64 image, do not repeat the base64 code.

Please use only the tools that are explicitly defined above.

**CRITICAL** When you don't need to use a tool, THEN ANSWER "<NO_TOOL_CALL>"
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

        gen_conf = dialog.llm_setting
        prompt_config = dialog.prompt_config
        dia_mcp_servers = gen_conf.get("mcp_servers")

        # 将每个mcp server的独有系统提示词附加到此次问答的系统提示词中
        system_prompt = prompt_config["system"]

        # 枚举当前对话助手所配置所有的mcp servers，生成tools desc
        mcp_instruction = self.mcp_instruction(dia_mcp_servers, current_user.email)
        mcp_messages = [{"role": "system", "content": mcp_instruction}]
        # mcp_messages.extend(mock_messages)
        mcp_messages.extend(messages)
        mock_messages = [
            {"role": "user", "content": "When you don't need to use a tool, THEN ANSWER '<NO_TOOL_CALL>"},
            {"role": "assistant", "content": "OK"},
        ]

        mcp_ans = ""
        # 强制关闭本地qwen3的思维链输出
        last_tool_call = ""
        msg_queue = queue.Queue()
        result_container = [None]  # 使用列表来共享结果，因为 nonlocal 在嵌套函数中可能有限制
        first_call = True
        while True:
            # 在提问前，要把mcp_messages复制一份再提问，因为chat会修改其内容
            tps = 0
            try:
                # 询问大模型，输出工具调用命令(当然也可能是最终回答)
                start_time = time.time()
                if first_call:
                    chat_msgs = deepcopy(mcp_messages[:-1])
                    chat_msgs.extend(mock_messages)
                    chat_msgs.append(mcp_messages[-1])
                    first_call = False
                else:
                    chat_msgs = deepcopy(mcp_messages)
                    chat_msgs.append(
                    {"role": "user", "content":
                     """<THINK>
1. 仔细分析是否还需要调用更多的工具才能回答用户的问题？
2. 如果答案是肯定的，那么选择一个工具并调用
3. 如果答案是否定的，那么直接返回<NO_TOOL_CALL>
</THINK>"""
                    }
                )

                mcp_gen_conf = {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 5,
                    "enable_cot": False,
                }
                logging.info(f"528- Ask {mcp_chat_mdl.llm_name}:\n{chat_msgs[-1]}")
                response_content = mcp_chat_mdl.chat(system_prompt, chat_msgs, mcp_gen_conf)
                end_time = time.time()
                duration = end_time - start_time
                tps = len(response_content) / duration if duration > 0 else 0

            except Exception as e:
                logging.error(f"406- **ERROR** {str(e)}")

            logging.info(f"411- {mcp_chat_mdl.llm_name}: %s", response_content)
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
                    result = await self.mcp_tool_call(current_user, mcp_server, tool_call, dialog, mcp_messages, chat_mdl, msg_queue)
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
                        msg = msg_queue.get(timeout=0.1)
                        if msg is None:
                            break  # 收到结束信号
                        yield {"answer": f"{mcp_ans}\n{msg}"}
                    except queue.Empty:
                        # 如果线程已经结束，则退出循环
                        if not thread.is_alive():
                            break

                # 等待线程结束
                thread.join()

                # 获取工具调用返回的最终结果
                result = result_container[0]
                result = result.replace(r'\\u', r'\u')

                # 去掉大模型在发出调用命令之前的思维链的内容
                response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)

                logging.info(f"492-工具执行结果：\n{result[:1024]}")

                # 将工具调用命令和结果附加到历史消息数组
                mcp_messages.append(
                    {"role": "assistant", "content": f"{response_content}\n\n工具执行结果：\n\n{result}" }
                )

                if dialog.description == 'DeepCoder':
                    # 如果是编程助手，则返回上传结果给用户
                    mcp_ans = f"{result}"
                    yield {"answer": mcp_ans}
                    break   # 退出会话
                elif "**UPLOAD**" in question:
                    # 如果是用户上传文件或者是编程助手，则返回上传结果给用户
                    mcp_ans += f"\n{result}"
                    yield {"answer": mcp_ans}
                    break   # 退出会话
                else:
                    # 提取```...```里面的内容
                    if len(result)>1024:
                        ignoreLen = len(result)-1024
                        result = result[:1024] + f"\n\n省略{ignoreLen}字..."

                    mcp_ans += f"\n\n工具执行结果:\n\n```\n{result}\n```\n\n"
                    yield {"answer": mcp_ans}
                    # 不退出会话,继续下一个循环,LLM会继续选择合适的工具，或者直接回答

            # 没有使用tool，表示已经收集了足够的信息，可以回答用户问题了
            else:
                last_msg = mcp_messages[-1]
                # 如果是调用了翻译pdf工具，就不需要重新问答了，直接返回翻译结果
                if ("<NO_TOOL_CALL>" in response_content or mcp_chat_mdl != chat_mdl) and "translate_pdf" not in last_msg["content"]:
                    final_prompt = system_prompt + "\n\n**CRITICAL**: REPLY DIRECTLY, DO NOT CALL TOOLS ANY MORE."

                    # 如果是调用了阅读文档工具，那么重新把用户的问题放到最后
                    if "read_document" in last_msg["content"]:
                        mcp_messages.append({"role": "user", "content": question})

                    # final_messages = [msg for msg in mcp_messages[1:] if "<NO_TOOL_CALL>" not in msg.get("content", "")]

                    # 用正式对话的模型重新问一次,由于第1条记录是mcp tools description
                    start_time = time.time()
                    for ans in chat_mdl.chat_streamly(final_prompt, mcp_messages[1:], gen_conf):
                        yield {"answer": f"{mcp_ans}\n{ans}"}

                    end_time = time.time()
                    duration = end_time - start_time
                    tps = len(ans) / duration if duration > 0 else 0
                    logging.info(f"639- {chat_mdl.llm_name}最终回答:\n{ans}")
                    response_content = ans

                if "ERROR" in response_content:
                    response_content += "\n\n**有错误发生，可能是因为上下文长度超限**"
                # mcp_ans += f"{response_content}\n\n*一共输出{len(response_content)}字，{round(tps, 2)}tokens/s*"

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
