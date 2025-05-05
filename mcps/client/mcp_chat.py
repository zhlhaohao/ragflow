from api.utils.log_utils import initRootLogger
initRootLogger("ragflow_server")

import asyncio
import json
import logging
from typing import Any
import jsonschema
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from api import settings
from mcps.client.lite_llm_json import LiteLLMJson
from api.db.services.llm_service import TenantLLMService, LLMBundle
from api.db import LLMType
from fastmcp import Client
from fastmcp.client.sampling import RequestContext, SamplingMessage, SamplingParams
import re

MCP_CHAT = None

async def init_mcp() -> None:
    global MCP_CHAT
    MCP_CHAT = McpChat()
    await MCP_CHAT.init_servers()


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
            return json.load(f)


class Server:
    """
    Manages MCP server connections and tool execution.
    """
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.tools: list[Any] = []

    def get_client(self, sampling_handler = None):
        config =  {"mcpServers": { self.name : self.config}}
        if sampling_handler:
            client = Client(config, sampling_handler = sampling_handler)
        else:
            client = Client(config)

        return client

    async def list_tools(self) -> list[Any]:
        tools = []
        client = self.get_client(self.name)
        async with client:
            logging.info(f"mcp server {self.name} 连接{client.is_connected()}")
            resp = await client.list_tools()
            for tool in resp:
                tools.append(Tool(tool.name, tool.description, tool.inputSchema))
        self.tools = tools
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        sampling_handler,
        retries: int = 2,
        delay: float = 2,
    ) -> Any:
        """带重试机制的调用工具.
        """
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                client = self.get_client(sampling_handler)

                # logger.info(f"89- 调用工具:{tool_name}, 参数是:{tool_args}\n")
                async with client:
                    result = await client.call_tool(tool_name, arguments)
                    # logger.info(f"101- 工具返回结果:\n{result}")
                    return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
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
Tool: {self.name}
Description: {self.description}
Arguments:
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
        server_config = config.load_config("conf/servers_config.json")

        self.servers = [
            Server(name, srv_config)
            for name, srv_config in server_config["mcpServers"].items()
        ]

        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)

        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        self.system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n"
            "If you need to access database,use a tool.\n"
            "If you are not clear about table name or table structure, use a tool.\n"
            "if the table does not exist,dont try to create a new table, just list tables of the database to find an appropriate table.\n\n"
            "CRITICAL: When you need to use a tool, you must ONLY Respond strictly in **JSON** and nothing else."
            " The response should adhere to the following JSON schema:\n"
            "## Response Format:\n"
            "{\n"
            '"tool": "string"\n'
            '"arguments": "dict"\n'
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above.\n"
        )

    async def process_llm_response(self, llm_response: str, sampling_handler) -> str:
        """分析llm的回答，如果需要则调用MCP工具.

        Args:
            llm_response: The response from the LLM.

        Returns:
            工具执行结果或者是入参
        """

        try:
            # 提取json string并解析出tool调用命令，如果不包含，那么抛出异常
            tool_call = llm_json.parse_response(llm_response)

            if "tool" in tool_call:
                if "arguments" not in tool_call:
                    tool_call["arguments"] = {}

                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                # tool_json = json.dumps(tool_call, ensure_ascii=False)
                # tool_desc = f"调用工具:\n\n```json\n{tool_json}\n```"


                for server in self.servers:
                    if any(tool.name == tool_call["tool"] for tool in server.tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"], sampling_handler
                            )

                            return f"\n\n工具执行结果:\n\n```\n{result}\n```"
                        except Exception as e:
                            error_msg = f"工具执行出错: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"找不到工具对应的MCP服务: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response
        except jsonschema.exceptions.ValidationError:
            return llm_response


    async def start(self) -> None:
        """
        Main chat session handler.
        """
        messages = [{"role": "system", "content": self.system_message}]

        while True:
            try:
                user_input = input("You: ").strip().lower()
                if user_input in ["quit", "exit"]:
                    logging.info("\nExiting...")
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
                    logging.info("\nAssistant: %s", response_content)

                    # 根据llm_response判断是否需要调用tool,并调用tool，然后返回结果
                    # 如果不使用tool，那么则原样返回

                    # 接收到工具中间结果输出
                    async def sampling_handler(
                        messages: list[SamplingMessage],
                        params: SamplingParams,
                        ctx: RequestContext,
                    ) -> str:
                        logging.info(f"\n{messages[0].content.text}")
                        return ""

                    result = await self.process_llm_response(response_content, sampling_handler)

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
                        logging.info("\nFinal response: %s", response_content)
                        messages.append(
                            {"role": "assistant", "content": response_content}
                        )
                        break

            except KeyboardInterrupt:
                logging.info("\nExiting...")
                break



    def chat(self, dialog, messages):
        """
        Main chat session handler.
        """
        llm_id, model_provider = TenantLLMService.split_model_name_and_factory(dialog.llm_id)
        # 从TenantLLM表取出模型信息（包括api_key）,然后封装成对象返回，也包装了chat_streamly和chat方法
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
        if not chat_mdl:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)

        prompt_config = dialog.prompt_config
        gen_conf = dialog.llm_setting
        mcp_messages = [{"role": "system", "content": self.system_message}]
        mcp_messages.extend(messages)
        ans = ""
        # 关闭本地qwen3的思维链输出
        gen_conf["extra_body"] = {"chat_template_kwargs":{"enable_thinking": False}}

        while True:
            # 第一步：询问llm，获得答案
            response_content = chat_mdl.chat(prompt_config["system"], mcp_messages, gen_conf)
            logging.info("\nAssistant: %s", response_content)

            try:
                tool_call = llm_json.parse_response(response_content)
                if "tool" in tool_call:
                    tool_json = json.dumps(tool_call, ensure_ascii=False)
                    ans  += f"调用工具:\n\n```json\n{tool_json}\n```\n\n"
                    yield {"answer": ans}
                    # 根据llm_response判断是否需要调用tool,并调用tool，然后返回结果
                    # 如果不使用tool，那么则原样返回
            except Exception as e:
                pass

            # 接收到工具中间结果输出
            async def sampling_handler(
                messages: list[SamplingMessage],
                params: SamplingParams,
                ctx: RequestContext,
            ):
                # yield {"answer": messages[0].content.text}
                print(f"{messages[0].content.text}")
                return ""

            result = asyncio.run(self.process_llm_response(response_content, sampling_handler))

            # 如果使用了tool
            if result != response_content:
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
                logging.info("\nFinal response: %s", response_content)
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
