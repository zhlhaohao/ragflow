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

MCP_CHAT = None

async def init_mcp() -> None:
    global MCP_CHAT
    MCP_CHAT = McpChat()
    await MCP_CHAT.init_servers()

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
    Manages MCP server connections and tool execution.
    """
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.tools: list[Any] = []

    def get_client(self, sampling_handler = None):
        if "command" in self.config:
            config =  {"mcpServers": { self.name : self.config}}
        if "url" in self.config:
            config =  {"mcpServers": {"url": self.config["url"]}}

        if sampling_handler:
            client = Client(config, sampling_handler = sampling_handler)
        else:
            client = Client(config)

        return client

    async def list_tools(self) -> list[Any]:
        tools = []
        client = self.get_client(self.name)
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
            # yield {"answer": messages[0].content.text}
            if msg_queue and messages[0].content.text:
                msg_queue.put(messages[0].content.text)
            # print(f"{messages[0].content.text}")
            return ""


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
    """Orchestrates the interaction between user, LLM, and tools."""

    async def init_servers(self):
        self.openai_client = OpenAI(
            base_url=settings.MCP_CHAT_URL,
            api_key=settings.MCP_CHAT_KEY,
        )

        config = Configuration()
        self.server_config = config.load_config("conf/mcp_config.toml")

        self.servers = [
            Server(name, srv_config)
            for name, srv_config in self.server_config["mcpServers"].items()
        ]

        self.server_tools = {}
        for server in self.servers:
            tools = await server.list_tools()
            self.server_tools[server.name] = tools


    def mcp_instruction(self, mcp_servers):
        tools_description = ''
        for server_name in mcp_servers:
            tools_description += f"\n\n## Tools of mcp server {server_name}:"
            tools = self.server_tools.get(server_name,[])
            tools_description += "\n".join([tool.format_for_llm() for tool in tools])

            config = self.server_config["mcpServers"].get(server_name,{})
            system_prompt = config.get("system")
            if system_prompt:
                tools_description += f"\n### Suggestion or extra information of mcp server {server_name}:\n{system_prompt}"

        today_desc = get_current_time_with_weekday()
        instruction = f"""
You are a helpful assistant with access to these tools:

{tools_description}

## Tool call guideline:
1. Choose the appropriate tool based on the user's question. If no tool is needed, reply directly
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
"""
        return instruction


    async def mcp_tool_call(self, server, tool_call, msg_queue = None) -> str:
        """分析llm的回答，如果需要则调用MCP工具.

        Args:
            llm_response: The response from the LLM.

        Returns:
            工具执行结果或者是入参
        """

        logging.info(f"279- Executing tool: {tool_call['tool']}")
        logging.info(f"280- With arguments: {tool_call['arguments']}")
        try:
            result = await server.execute_tool(
                tool_call["tool"], tool_call["arguments"], msg_queue
            )
            if "data:image" in result:
                return f"\n\n{result}"
            else:
                result = self.convert_mixed_utf_string(result)
                return f"\n\n工具执行结果:\n\n```\n{result}\n```"
        except Exception as e:
            error_msg = f"291- 工具执行出错: {str(e)}"
            logging.error(error_msg)
            return error_msg


    async def start(self) -> None:
        """
        Main chat session handler.
        """
        messages = [{"role": "system", "content": self.system_message}]

        while True:
            try:
                user_input = input("You: ").strip().lower()
                if user_input in ["quit", "exit"]:
                    logging.info("\n306- Exiting...")
                    break

                # 导入用户的问题
                messages.append({"role": "user", "content": user_input})

                while True:
                    # 第一步：询问llm，获得答案
                    response = self.openai_client.chat.completions.create(
                        model = settings.MCP_CHAT_MDL,
                        messages = messages,
                        temperature = 0.7,
                        max_tokens = 4096,
                        top_p = 1.0,
                        stream = False,
                        stop = None,
                    )
                    response_content = response.choices[0].message.content
                    logging.info("\n324- Assistant: %s", response_content)
                    result = await self.mcp_tool_call(response_content)

                    # 如果使用了tool
                    if result != response_content:
                        # 将tool的调用结果加入到历史信息中
                        messages.append(
                            {"role": "assistant", "content": response_content}
                        )
                        messages.append({"role": "system", "content": result})

                        # 循环调用llm，获取最终的回复
                        continue
                    # 没有使用tool
                    else:
                        logging.info("\n339- Final response: %s", response_content)
                        messages.append(
                            {"role": "assistant", "content": response_content}
                        )
                        break

            except KeyboardInterrupt:
                logging.info("\n346- Exiting...")
                break

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

    def chat(self, dialog, messages):
        """
        Main chat session handler.
        """
        llm_id, model_provider = TenantLLMService.split_model_name_and_factory(dialog.llm_id)
        # 从TenantLLM表取出模型信息（包括api_key）,然后封装成对象返回，也包装了chat_streamly和chat方法
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
        if not chat_mdl:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)

        gen_conf = dialog.llm_setting
        prompt_config = dialog.prompt_config
        dia_mcp_servers = gen_conf.get("mcp_servers")

        # 将每个mcp server的独有系统提示词附加到此次问答的系统提示词中
        system_prompt = prompt_config["system"]
        # for server_name in dia_mcp_servers:
        #     config = self.server_config["mcpServers"].get(server_name,{})
        #     if config.get("system"):
        #         system_prompt += "\n\n" + config.get("system")

        # 枚举当前对话助手所配置所有的mcp servers，生成tools desc
        mcp_instruction = self.mcp_instruction(dia_mcp_servers)
        mcp_messages = [{"role": "system", "content": mcp_instruction}]
        mcp_messages.extend(messages)

        ans = ""
        # 强制关闭本地qwen3的思维链输出
        gen_conf["enable_cot"] = False

        msg_queue = queue.Queue()
        result_container = [None]  # 使用列表来共享结果，因为 nonlocal 在嵌套函数中可能有限制
        while True:
            # 第一步：询问llm，获得答案
            response_content = chat_mdl.chat(system_prompt, mcp_messages, gen_conf)
            logging.info("\n411- Assistant: %s", response_content)

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
                            ans  += f"调用MCP插件{server.name}:\n\n```json\n{tool_json}\n```\n\n"
                            yield {"answer": ans}
            except Exception as ex:
                pass
                # error_msg = f"parse_response error: {str(ex)}"
                # logging.error(error_msg)

            # result = asyncio.run(self.mcp_tool_call(response_content, sampling_handler))

            # 如果要求使用tool
            if mcp_server is not None:
                async def run_async_func():
                    # 执行异步函数B并获取结果
                    result = await self.mcp_tool_call(mcp_server, tool_call, msg_queue)
                    result_container[0] = result
                    # 执行结束标记
                    msg_queue.put(None)

                def start_event_loop():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(run_async_func())


                # 启动异步事件循环的线程
                thread = threading.Thread(target=start_event_loop)
                thread.start()

                while True:
                    try:
                        msg = msg_queue.get(timeout=0.1)
                        if msg is None:
                            break  # 收到结束信号
                        yield {"answer": f"{ans}, {msg}"}
                    except queue.Empty:
                        if not thread.is_alive():
                            break

                thread.join()
                result = result_container[0]
                result = result.replace(r'\\u', r'\u')

                # 去掉思维链的内容
                response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)

                # 将tool的调用结果加入到历史信息中
                mcp_messages.append(
                    {"role": "assistant", "content": response_content}
                )
                mcp_messages.append({"role": "system", "content": result})

                # 循环调用llm，获取最终的回复
                # if '<think>' not in ans:
                #     ans += '<think>'
                ans += result + "\n\n"
                yield {"answer": ans}

            # 没有使用tool，表示是最终回答
            else:
                logging.info("\n486- Final response: %s", response_content)
                # if '<think>' in ans and '</think>' not in ans:
                #     ans += '</think>'
                ans += response_content

                yield {"answer": ans}
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
