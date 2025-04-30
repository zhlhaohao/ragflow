from api.utils.log_utils import initRootLogger
initRootLogger("ragflow_server")

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import jsonschema
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anyio

from pydantic import BaseModel
from openai import OpenAI
from api import settings
from mcps.client.lite_llm_json import LiteLLMJson
from api.db.services.llm_service import LLMService, TenantLLMService, LLMBundle
from api.db import LLMType

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
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.tools: list[Any] = []
        self.closed = True

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        self.tools = tools
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 0.1,
    ) -> Any:
        """带重试机制的调用工具.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                await self.initialize()
                result = await self.session.call_tool(tool_name, arguments)
                return result

            except anyio.ClosedResourceError as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                await self.initialize()
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    # await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

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

    async def cleanup(self) -> None:
        """
        清理MCP服务器资源
        """
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


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

        self.servers = []
        try:
            config = Configuration()
            server_config = config.load_config("conf/servers_config.json")

            self.servers = [
                Server(name, srv_config)
                for name, srv_config in server_config["mcpServers"].items()
            ]

            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

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
        except Exception as ex:
            await self.cleanup_servers()


    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        logging.info("cleanup servers...")
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        self.servers = []
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")



    async def process_llm_response(self, llm_response: str) -> str:
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

                tool_json = json.dumps(tool_call, ensure_ascii=False)
                tool_desc = f"调用工具:\n\n```json\n{tool_json}\n```"

                for server in self.servers:
                    if any(tool.name == tool_call["tool"] for tool in server.tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"进度: {progress}/{total} ({percentage:.1f}%)"
                                )

                            return f"{tool_desc}\n\n工具执行结果:\n\n```\n{result}\n```"
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
                    result = await self.process_llm_response(response_content)

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

        #     answer = ""
        #     for ans in chat_mdl.chat_streamly(prompt, messages, gen_conf):
        #         answer = ans
        #         yield {"answer": answer}

        # else:
        #     answer = chat_mdl.chat(prompt_config["system"], messages, gen_conf)
        #     yield answer

        mcp_messages = [{"role": "system", "content": self.system_message}]
        mcp_messages.extend(messages)
        ans = ""
        # 关闭qwen3的思维链输出
        gen_conf["extra_body"] = {"chat_template_kwargs":{"enable_thinking": False}}

        while True:
            # 第一步：询问llm，获得答案
            response_content = chat_mdl.chat(prompt_config["system"], mcp_messages, gen_conf)

            # response = self.openai_client.chat.completions.create(
            #     model = settings.MCP_CHAT_MDL,
            #     messages = mcp_messages,
            #     temperature = 0.7,
            #     max_tokens = 4096,
            #     top_p = 1.0,
            #     stream = False,
            #     stop = None,
            # )
            # response_content = response.choices[0].message.content
            logging.info("\nAssistant: %s", response_content)

            # 根据llm_response判断是否需要调用tool,并调用tool，然后返回结果
            # 如果不使用tool，那么则原样返回
            result = asyncio.run(self.process_llm_response(response_content))

            # 如果使用了tool
            if result != response_content:
                # 将tool的调用结果加入到历史信息中
                mcp_messages.append(
                    {"role": "assistant", "content": response_content}
                )
                mcp_messages.append({"role": "system", "content": result})

                # 循环调用llm，获取最终的回复
                if '<think>' not in ans:
                    ans += '<think>'
                ans += result + "\n\n"
                yield {"answer": ans}
            # 没有使用tool，表示是最终回答
            else:
                logging.info("\nFinal response: %s", response_content)
                if '<think>' in ans and '</think>' not in ans:
                    ans += '</think>'
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
    await mcp_chat.cleanup_servers()

if __name__ == "__main__":

    asyncio.run(main())
