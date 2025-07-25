
import os
import time
import jwt
import requests
import json

## 生成token
def generate_token(app_key: str, app_secret: str, exp_seconds: int):

    now = int(round(time.time()))
    payload = {
        "app_key": app_key,
        "exp": now + exp_seconds,
        "iat": now
    }
    return jwt.encode(
        payload,
        app_secret,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"}
    )

## 非流式调用
def completions(app_key: str, app_secret: str,exp_seconds: int, model: str,messages:list,**kwargs:dict):
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
        if os.environ.get("UNIIN_BASE_URL"):
            endpoint = os.environ.get("UNIIN_BASE_URL")

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
    )

    headers = {
    'Content-Type': 'application/json',
    'Authorization': token
    }
    url = endpoint+"/openapi/v1/chat/completions"
    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.post(url, headers=headers, data=payload, proxies=proxies)
    # print(response.text)
    return response.text

## 流式调用
def stream_completions(app_key: str, app_secret: str,exp_seconds: int,  model_name: str,messages:list,**kwargs:dict):
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
        if os.environ.get("UNIIN_BASE_URL"):
            endpoint = os.environ.get("UNIIN_BASE_URL")

    payload = json.dumps(
        {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **kwargs
        }
    )

    headers = {
    'Content-Type': 'application/json',
    'Authorization': token,
    'Accept':'text/event-stream'
    }
    url = endpoint+"/openapi/v1/chat/completions"
    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.post(url, headers=headers, data=payload, stream=True, proxies=proxies)
    return response


## Prompt内容获取
def get_prompt(app_key: str, app_secret: str,exp_seconds:int, prompt_id: int,**kwargs:dict):

    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
    version = kwargs.get("version",None)
    if version is not None:
        url = endpoint+"/openapi/v1/prompt-template"+f"/{prompt_id}"+f"?versionNum={version}"
    else:
        url = endpoint+"/openapi/v1/prompt-template"+f"/{prompt_id}"
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    payload = {}
    headers = {
        'Authorization': token
        }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.text
    # print(response.text)

## 替换Prompt变量内容
def get_replaced_content(app_key: str, app_secret: str,exp_seconds:int,prompt_id:int,variables:dict,**kwargs:dict):

    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
    version = kwargs.get("version",None)
    if version is not None:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"+f"?versionNum={version}"
    else:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    payload = json.dumps(
        {
            "variables":variables
        }
    )
    headers = {
    'Content-Type': 'application/json',
    'Authorization': token
    }

    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.request("POST", url, headers=headers, data=payload, proxies=proxies)

    # print(response.text)
    return response.text

## 使用Prompt进行大模型非流式调用
def completions_with_prompt(app_key: str, app_secret: str,exp_seconds: int, prompt_id:int,variables:dict,model: str,**kwargs:dict):

    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
    version = kwargs.get("version",None)
    if version is not None:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"+f"?versionNum={version}"
    else:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    payload = json.dumps(
        {
            "variables":variables
        }
    )
    headers = {
    'Content-Type': 'application/json',
    'Authorization': token
    }

    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.request("POST", url, headers=headers, data=payload, proxies=proxies)

    replacedContent = json.loads(response.text)['data']['replacedContent']
    #### 非流式调用
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": replacedContent
                }
            ],
            "stream": False,
            **kwargs
        }
    )

    headers = {
    'Content-Type': 'application/json',
    'Authorization': token
    }
    completions_url = endpoint+"/openapi/v1/chat/completions"

    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.post(completions_url, headers=headers, data=payload, proxies=proxies)
    return response.text


## 使用Prompt进行大模型流式调用
def stream_completions_with_prompt(app_key: str, app_secret: str,exp_seconds: int, prompt_id:int,variables:dict,model: str,**kwargs:dict):

    endpoint = kwargs.get("endpoint",None)
    if endpoint is not None:
        kwargs.pop("endpoint")
    else:
        endpoint = "https://openai.uniin.cn"
    version = kwargs.get("version",None)
    if version is not None:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"+f"?versionNum={version}"
    else:
        url = endpoint+"/openapi/v1/prompt-template/replaced"+f"/{prompt_id}"
    token = generate_token(app_key=app_key, app_secret=app_secret,exp_seconds=exp_seconds)
    payload = json.dumps(
        {
            "variables":variables
        }
    )
    headers = {
    'Content-Type': 'application/json',
    'Authorization': token
    }

    proxy = os.environ.get("UNIIN_PROXY")
    proxies = {
        'http': proxy,
        'https': proxy,
    } if proxy else None
    response = requests.request("POST", url, headers=headers, data=payload, proxies=proxies)

    #### 非流式调用
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": response.text
                }
            ],
            "stream": False,
            **kwargs
        }
    )

    headers = {
    'Content-Type': 'application/json',
    'Authorization': token,
    'Accept':'text/event-stream'
    }
    completions_url = endpoint+"/openapi/v1/chat/completions"
    response = requests.post(completions_url, headers=headers, data=payload,stream=True, proxies=proxies)
    return response
