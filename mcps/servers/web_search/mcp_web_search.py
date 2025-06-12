from fastmcp import FastMCP, Context
import aiohttp
import requests
from openai import AsyncOpenAI
import logging
import argparse
import asyncio
import yaml
import json

mcp_server = FastMCP("search")

# 读取配置文件
with open("conf/local.service_conf.yaml", "r") as f:
    config = yaml.safe_load(f)

llm_config = config.get('user_default_llm')
base_url = llm_config.get('mcp_chat_url')
api_key = llm_config.get('mcp_chat_key')
model_name = llm_config.get('mcp_chat_model')
jina_api_key = llm_config.get('jina_api_key')

total_pages = 0

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 创建文件处理器
file_handler = logging.FileHandler("test.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

client = AsyncOpenAI(
    base_url = base_url,
    api_key = api_key,
)


async def generate_query(query, stream=False):
    """
    将问题生成N个不同的问题
    """
    prompt = f"""You are an expert research assistant. Given the user's query, generate up to {args.query_count} distinct, precise search queries in chinese that would help gather comprehensive information on the topic.
    Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise research assistant.",
            },
            {"role": "user", "content": f"User Query: {query}\n\n{prompt}"},
        ],
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}},
    )
    logger.info(
        f"新生成的{args.query_count}个查询词: {response.choices[0].message.content}"
    )

    return response.choices[0].message.content


async def if_useful(query: str, page_text: str):
    prompt = """You are a critical research evaluator. Given the user's query and the content of a webpage, determine if the webpage contains information relevant and useful for addressing the query.
    Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a strict and concise evaluator of research relevance.",
            },
            {
                "role": "user",
                "content": f"User Query: {query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}",
            },
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    response = response.choices[0].message.content

    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            # Fallback: try to extract Yes/No from the response.
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"


# 返回网页内容上与问题有关的片段
async def extract_relevant_context(query, search_query, page_text):
    prompt = f"""你是一位专业的信息提取专家。根据用户查询从网页内容中提取和摘要出对回答用户查询有帮助的相关信息。只返回相关的上下文作为纯文本，最多{args.context_length}字，不添加任何评论,如何你发现该网页并不能解答用户的问题，请直接返回：网页内容无关。"""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一个摘要专家",
            },
            {
                "role": "user",
                "content": f"用户查询: {query}\n搜索关键词: {search_query}\n\n网页内容:\n{page_text[:20000]}\n\n{prompt}",
            },
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    response = response.choices[0].message.content
    if response:
        return response.strip()
    return ""


async def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = """You are an analytical research assistant. Based on the original query, the search queries performed so far, and the extracted contexts from webpages, determine if further research is needed.
    If further research is needed, provide up to four new search queries as a Python list (for example, ['new query1', 'new query2']). If you believe no further research is needed, respond with exactly .
    Output only a Python list or the token  without any additional text."""

    content = f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"
    # logger.info(f"content: {content}")
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in extracting and summarizing relevant information.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    response = response.choices[0].message.content
    if response:
        cleaned = response.strip()
        if cleaned == "" or cleaned == "[]":
            return ""
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                logger.info(f"new_queries:{new_queries}")
                return new_queries
            else:
                logger.info(
                    f"LLM did not return a list for new search queries. Response: {response}"
                )
                return []
        except Exception as e:
            logger.error(f"Error parsing new search queries:{e}, Response:{response}")
            return []
    return []


async def searxng_search(query: str):
    """通过searxng在互联网异步搜索用户的问题，返回前web_search个url
    http://127.0.0.1:8088/search?format=json&q=广州天气&language=zh-CN&time_range=&safesearch=0&categories=general
    http://10.119.101.20:9860/search?format=json&q=广州天气&language=zh-CN&time_range=&safesearch=0&categories=general


    """
    logger.info(f"searxng_search:{query}")
    links = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.get(
                f"{args.searxng_url}search?format=json&q={query}&language=zh-CN&time_range=&safesearch=0&categories=general"
            ) as response:
                results = (await response.json())["results"]
                logger.info(f"searxng_search results:{results}")
                links = [result["url"] for result in results[: args.max_results]]
    except Exception as e:
        logger.error(f"Web search error: {e}")
    return links


async def fetch_webpage_text(url, ctx):
    """Firecrawl爬取网页的内容

curl -X POST http://10.119.101.21:9860/v1/scrape \
    -H 'Content-Type: application/json' \
    -d '{
      "url": "https://www.zaobao.com/news/china/story20250513-6328924",
      "onlyMainContent": true,
      "formats" : ["markdown"]
    }'

    Args:
        url (_type_): url

    Returns:
        _type_: _description_
    """
    full_url = f"{args.firecrawl_url}v1/scrape"

    headers = {
        "Content-Type": "application/json",
    }
    param = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": 2000,
        "timeout": 30000,
    }

    global total_pages
    try:
        logger.info("开始爬取")
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(full_url, json=param, timeout=30) as resp:
                logger.info("爬取结束")
                total_pages += 1
                await ctx.sample(f"🌍 {total_pages}已爬取")
                if resp.status == 200:
                    resp = await resp.text()
                    result = json.loads(resp)
                    content = result.get("data").get("markdown")
                    # logger.info(f"content:{content}")
                    return content
                else:
                    text = await resp.text()
                    logger.info(
                        f"207- Firecrawl爬取 {url} 失败: {resp.status} - {text}"
                    )
                    return None
    except Exception as e:
        logger.error(f"210-Error fetching webpage text with Firecrawl:{e}")
        return None


async def process_link(link, query, search_query, ctx):
    """firecrawl爬取网页内容，然后提取与问题相关的200个字

    Args:
        link (_type_): 网页url
        query (_type_): 用户的提问
        search_query (_type_): 用户提问整理后的搜索词

    Returns:
        _type_: 返回网页上与用户提问相关的片段(200字符)
    """
    page_text = None
    if link:  # and not link.endswith(".pdf"):
        logger.info(f"爬取网页内容: {link}")
        page_text = await fetch_webpage_text(link, ctx)

    if page_text is None:
        return None

    # # 判断内容是否能够解答问题
    # if args.deep_research:
    #     usefulness = await if_useful(query, page_text)
    #     logger.info(f"网页是否能够解答问题: {usefulness}")
    # else:
    #     usefulness = "Yes"

    # 提取网页内容上与用户提问相关的片段
    # if usefulness == "Yes":
    logger.info("提炼摘要")
    context = await extract_relevant_context(query, search_query, page_text)
    if context:
        await ctx.sample(f"摘要:\n{context}\n\n")
        return context
    return None


async def get_images_description(iamge_url):
    completion = await client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "使用一句话描述图片的内容"},
                    {"type": "image_url", "image_url": {"url": iamge_url}},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


@mcp_server.tool()
async def web_search(query: str, ctx: Context) -> str:
    """互联网搜索用户的问题,返回搜索结果

    Args:
        query (str): 用户问题

    Returns:
        str: 查询结果,多个结果用\n\n分隔
    """
    # 反复询问大模型已经搜索的资料是否足够,如果不够那么循环,最多循环多少次
    iteration_limit = args.iteration_limit
    iteration = 0
    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    # logger.info(f"234- Searching for: {query}")

    try:
        # 让大模型将用户的提问扩展为N个不同的问题
        if args.deep_research:
            new_search_queries = eval(await generate_query(query))
            all_search_queries.extend(new_search_queries)
        else:
            new_search_queries = all_search_queries = [query]

        while iteration < iteration_limit:
            logger.info(f"\n=== 第{iteration + 1}次循环 ===")
            # await ctx.sample(f"\n=== 第{iteration + 1}次循环 ===")

            iteration_contexts = []

            # 调用searxng对4个扩展问题进行搜索，每个问题取前2个搜索结果，形成一个url数组
            # 这里可以并发处理
            # search_tasks = [searxng_search(query) for query in new_search_queries]
            # search_results = await asyncio.gather(*search_tasks)
            search_results = [await searxng_search(query) for query in new_search_queries]

            # 结果去重
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            await ctx.sample(f"共搜索出{len(unique_links)}个网页.")

            # link_results = [
            #     await process_link(link, query, unique_links[link], ctx) for link in unique_links
            # ]

            # firecrawl爬取url的内容，询问大模型词网页是否有用，有用则返回与用户提问相关的片段
            # 创建信号量限制并发数为3,一个批次启动3个任务
            semaphore = asyncio.Semaphore(5)

            async def process_link_with_sem(link, query, search_query, ctx):
                async with semaphore:
                    return await process_link(link, query, search_query, ctx)

            # 每个批次的任务间隔1秒启动
            async def delayed_task(index, link):
                await asyncio.sleep(index * 1)
                return await process_link_with_sem(link, query, unique_links[link], ctx)

            # 创建所有任务批次,开始执行
            tasks = [
                asyncio.wait_for(delayed_task(i, link), timeout=120)
                for i, link in enumerate(unique_links)
            ]
            # 收集结果
            link_results = await asyncio.gather(*tasks, return_exceptions=True)
            link_results = [
                res for res in link_results if not isinstance(res, Exception)
            ]

            # 去掉None值
            i = 0
            for res in link_results:
                if res:
                    await ctx.sample(f"ID{i}: {res[0:100]}\n\n")
                    iteration_contexts.append(res)
                    i += 1

            # 累加结果
            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                logger.info("此次循环没找到相关内容.")

            if args.deep_research:
                # 询问大模型是否获取的资料已经足够，是否还需要再次循环搜索
                new_search_queries = await get_new_search_queries(
                    query, all_search_queries, aggregated_contexts
                )

                if new_search_queries == "":
                    logger.info("资料已经收集完成，结束搜索\n\n")
                    break
                elif len(new_search_queries) > 0:
                    # 大模型说还不够，然后给出了新的问题
                    logger.info(
                        f"由于对结果不满意，LLM提供了新的问题:{new_search_queries}"
                    )
                    all_search_queries.extend(new_search_queries)
                else:
                    # logger.info("LLM说可以结束搜索了.")
                    ctx.info("资料已经收集完成，结束搜索\n\n")
                    break

            iteration += 1
        ans = "\n\n".join(aggregated_contexts)
        await ctx.sample(f"从网上得到{len(ans)}字的回答: {ans[0:100]}...")
        return ans

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return "An error occurred while processing the query."


@mcp_server.tool()
async def get_images(query: str) -> str:
    """获取图片链接和描述"""
    logger.info(f"Searching for images for query: {query}")
    response = requests.get(
        f"{args.searxng_url}search?format=json&q={query}&language=zh-CN&time_range=month&safesearch=0&categories=images"
    )
    results = response.json()["results"]
    img_srcs = []
    for result in results[:2]:
        img_srcs.append(result["img_src"])

    result = {}

    for img_src in img_srcs:
        logger.info(f"Fetching image description for: {img_src}")
        description = await get_images_description(img_src)
        logger.info(f"Image description for {img_src}: {description}")
        result[img_src] = description

    return result


args = {}
if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="Web Search Server")

    parser.add_argument(
        "--deep-research",
        action="store_true",  # This will set the value to True if the flag is present
        default=False,  # Default value when the flag is not provided
        help="true 耗时较长",
    )

    parser.add_argument(
        "--iteration-limit",
        default="1",
        type=int,
        help="如果大模型说资料还不够的话，最多N轮次的循环搜索 ",
    )
    parser.add_argument(
        "--query-count",
        default="1",
        type=int,
        help="将用户问题扩展成N个问题",
    )
    parser.add_argument(
        "--max-results",
        default="6",
        type=int,
        help="每次搜索取前N个结果",
    )
    parser.add_argument(
        "--context-length",
        default="1000",
        type=int,
        help="提取网页上与问题相关的片段最多N个字符",
    )
    parser.add_argument(
        "--firecrawl-url",
        default="http://10.119.101.21:9860/",
        type=str,
        help="firecrawl网址",
    )
    parser.add_argument(
        "--searxng-url",
        default="http://127.0.0.1:8088/",
        type=str,
        help="searxng网址",
    )
    # parser.add_argument(
    #     "--http-proxy",
    #     default=None,
    #     type=str,
    #     help="jina爬虫代理地址",
    # )
    args = parser.parse_args()

    logger.info(f"Starting web search MCP,args:{args}")
    mcp_server.run()
    # mcp_server.run(transport="sse", host="0.0.0.0", port=8000)
