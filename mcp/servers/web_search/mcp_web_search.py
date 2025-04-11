from mcp.server.fastmcp import FastMCP

# import pymysql
import requests
from openai import OpenAI

# import pandas as pd
import logging
import argparse

mcp = FastMCP("search")


base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-db2ad92210a348fd884c3b94655095c5"
model_name = "deepseek-v3"

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

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


def generate_query(query, stream=False):
    """
    将问题生成N个不同的问题
    """
    query_count = ["one", "two", "three", "four", "five", "six"][args.query_count - 1]
    prompt = f"""You are an expert research assistant. Given the user's query, generate up to {query_count} distinct, precise search queries in chinese that would help gather comprehensive information on the topic.
    Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise research assistant.",
            },
            {"role": "user", "content": f"User Query: {query}\n\n{prompt}"},
        ],
    )
    logger.info(
        f"新生成的{args.query_count}个查询词: {response.choices[0].message.content}"
    )

    return response.choices[0].message.content


def if_useful(query: str, page_text: str):
    prompt = """You are a critical research evaluator. Given the user's query and the content of a webpage, determine if the webpage contains information relevant and useful for addressing the query.
    Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."""

    response = client.chat.completions.create(
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
def extract_relevant_context(query, search_query, page_text):
    prompt = """You are an expert information extractor. Given the user's query, the search query that led to this page, and the webpage content, extract all pieces of information that are relevant to answering the user's query.
    Return only the relevant context as plain text without commentary."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in extracting and summarizing relevant information.",
            },
            {
                "role": "user",
                "content": f"User Query: {query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}",
            },
        ],
    )

    response = response.choices[0].message.content
    if response:
        return response.strip()
    return ""


def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = """You are an analytical research assistant. Based on the original query, the search queries performed so far, and the extracted contexts from webpages, determine if further research is needed.
    If further research is needed, provide up to four new search queries as a Python list (for example, ['new query1', 'new query2']). If you believe no further research is needed, respond with exactly .
    Output only a Python list or the token  without any additional text."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in extracting and summarizing relevant information.",
            },
            {
                "role": "user",
                "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}",
            },
        ],
    )

    response = response.choices[0].message.content
    if response:
        cleaned = response.strip()
        if cleaned == "":
            return ""
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
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


def web_search(
    query: str,
):
    """通过searxng在互联网搜索用户的问题，返回前2个url

    Args:
        query (str): 用户问题

    Returns:
        List: url数组
    """
    links = []
    # logger.info(f"164- Searching for: {query}")
    response = requests.get(
        f"{args.searxng_url}search?format=json&q={query}&language=zh-CN&time_range=&safesearch=0&categories=general",
        timeout=30,
    )
    results = response.json()["results"]
    for result in results[: args.max_results]:
        links.append(result["url"])

    return links


def fetch_webpage_text(url):
    """Jina爬取网页的内容

    Args:
        url (_type_): url

    Returns:
        _type_: _description_
    """
    JINA_BASE_URL = args.jina_url
    full_url = f"{JINA_BASE_URL}{url}"

    headers = {
        "X-Respond-With": "markdown",
        "X-With-Generated-Alt": "true",
        "X-User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "X-Timeout": "60",
    }
    if args.http_proxy is not None:
        headers["X-Proxy-Url"] = args.http_proxy

    try:
        resp = requests.get(full_url, headers=headers, timeout=120)
        if resp.status_code == 200:
            return resp.text
        else:
            text = resp.text
            logger.info(f"207- Jina爬取 {url} 失败: {resp.status_code} - {text}")
            return ""
    except Exception as e:
        logger.error(f"210-Error fetching webpage text with Jina:{e}")
        return ""


def process_link(link, query, search_query):
    """_summary_

    Args:
        link (_type_): 网页url
        query (_type_): 用户的提问
        search_query (_type_): 用户提问整理后的搜索词

    Returns:
        _type_: 返回网页上与用户提问相关的片段(200字符)
    """
    logger.info(f"爬取网页内容: {link}")
    page_text = fetch_webpage_text(link)
    if not page_text:
        return None

    # 判断内容是否能够解答问题
    usefulness = if_useful(query, page_text)
    logger.info(f"网页是否能够解答问题: {usefulness}")
    if usefulness == "Yes":
        # 提取网页内容上与用户提问相关的片段
        context = extract_relevant_context(query, search_query, page_text)
        # 返回200个字符作为上下文
        if context:
            logger.info(
                f"Extracted context from {link} (first 200 chars): {context[: args.context_length]}"
            )
            return context
    return None


def get_images_description(iamge_url):
    completion = client.chat.completions.create(
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


@mcp.tool()
def search(query: str) -> str:
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

    # 让大模型将用户的提问扩展为4个不同的问题
    new_search_queries = eval(generate_query(query))
    all_search_queries.extend(new_search_queries)

    while iteration < iteration_limit:
        logger.info(f"\n=== 第{iteration + 1}次循环 ===")
        iteration_contexts = []

        # 调用searxng对4个扩展问题进行搜索，每个问题取前2个搜索结果，形成一个url数组
        search_results = [web_search(query) for query in new_search_queries]

        # unique_links是所有查询出来的web links
        unique_links = {}
        for idx, links in enumerate(search_results):
            query = new_search_queries[idx]
            for link in links:
                if link not in unique_links:
                    unique_links[link] = query

        logger.info(f"本次循环共搜索出{len(unique_links)}个url.")

        # 爬取url的内容，返回与用户提问相关的片段（前200字符）
        link_results = [
            process_link(link, query, unique_links[link]) for link in unique_links
        ]

        # 去掉None值
        for res in link_results:
            if res:
                iteration_contexts.append(res)

        # 累加结果
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            logger.info("No useful contexts were found in this iteration.")

        # 询问大模型是否获取的资料已经足够，是否还需要再次循环搜索
        new_search_queries = get_new_search_queries(
            query, all_search_queries, aggregated_contexts
        )

        if new_search_queries == "":
            logger.info("LLM indicated that no further research is needed.")
            break
        elif new_search_queries:
            # 大模型说还不够，然后给出了新的问题
            logger.info(f"LLM provided new search queries:{new_search_queries}")
            all_search_queries.extend(new_search_queries)
        else:
            logger.info("LLM说可以结束搜索了.")
            break

        iteration += 1
    return "\n\n".join(aggregated_contexts)


@mcp.tool()
def get_images(query: str) -> str:
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
        description = get_images_description(img_src)
        logger.info(f"Image description for {img_src}: {description}")
        result[img_src] = description

    return result


args = {}
if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="Web Search Server")
    parser.add_argument(
        "--iteration-limit",
        default="4",
        type=int,
        help="如果大模型说资料还不够的话，最多N轮次的循环搜索 ",
    )
    parser.add_argument(
        "--query-count",
        default="4",
        type=int,
        help="将用户问题扩展成N个问题",
    )
    parser.add_argument(
        "--max-results",
        default="2",
        type=int,
        help="每次搜索取前N个结果",
    )
    parser.add_argument(
        "--context-length",
        default="200",
        type=int,
        help="提取网页上与问题相关的片段最多N个字符",
    )
    parser.add_argument(
        "--jina-url",
        default="https://r.jina.ai/",
        type=str,
        help="jina reader网址",
    )
    parser.add_argument(
        "--searxng-url",
        default="http://127.0.0.1:8088/",
        type=str,
        help="searxng网址",
    )
    parser.add_argument(
        "--http-proxy",
        default=None,
        type=str,
        help="jina爬虫代理地址",
    )

    args = parser.parse_args()

    logger.info(f"Starting web search MCP,args:{args}")
    mcp.run()
