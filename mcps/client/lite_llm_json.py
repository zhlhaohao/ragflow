import json
import re
from typing import Dict
from json_repair import repair_json
import jsonschema
from jsonschema.exceptions import SchemaError, ValidationError


class LiteLLMJson:
    BASE_PROMPT = """{query_prompt}

## Response Format:

Respond strictly in **JSON**. The response should adhere to the following JSON schema:

```JSON schema
{json_schema}
```

"""

    def __init__(self, json_schema: dict) -> None:
        """
        Initialize the class with a JSON schema.

        Args:
            json_schema (dict): The JSON schema.

        Returns:
            None
        """

        # Validate the JSON schema
        try:
            jsonschema.validate({}, json_schema)
        except SchemaError as e:
            raise e
        except ValidationError:
            pass

        # Set the JSON schema attribute
        self.json_schema = json_schema

    def generate_prompt(self, query_prompt: str) -> str:
        """
        Generate a prompt for the given query prompt.

        Args:
            query_prompt (str): The query prompt.

        Returns:
            str: The generated prompt.
        """
        # Convert the JSON schema to a string
        json_schema_str = json.dumps(self.json_schema, ensure_ascii=False, indent=2)

        # Format the prompt string with the query prompt and JSON schema
        return self.BASE_PROMPT.format(
            query_prompt=query_prompt, json_schema=json_schema_str
        )

    def parse_response(self, response: str) -> Dict:
        """
        Parse the response string and validate it against the JSON schema.

        Args:
            response (str): The response string to parse.

        Returns:
            Dict: The parsed response as a dictionary.

        """

        json_data = self._extract_data_from_response(response)
        jsonschema.validate(json_data, self.json_schema)

        return json_data

    def _extract_data_from_response(
        self, text: str, decoder=json.JSONDecoder(strict=False), symbols=("{", "[")
    ):
        """从响应文本中提取并验证JSON数据"""
        pos = 0  # 当前搜索起始位置
        last_exception = None  # 记录最后一次验证异常

        # 循环查找可能的JSON起始符号{[
        while True:
            # 查找所有符号在文本中的位置
            matches = {s: text.find(s, pos) for s in symbols}
            # 过滤未找到的符号(-1)
            matches = {k: v for k, v in matches.items() if v != -1}
            # 如果找不到任何符号则退出循环
            if not matches:
                break

            # 获取最早出现的符号及其位置
            symbol, match_pos = min(matches.items(), key=lambda x: x[1])
            try:
                # 尝试从该位置解析JSON
                result, index = decoder.raw_decode(
                    repair_json(text[match_pos:], ensure_ascii=False)
                )
                # 确保结果可序列化
                json_result = json.loads(json.dumps(result))

                # 立即进行schema校验
                try:
                    jsonschema.validate(json_result, self.json_schema)
                    return json_result  # 验证成功直接返回
                except ValidationError as e:
                    last_exception = e  # 记录验证异常
                    pos = match_pos + index  # 向后移动位置继续查找
            except ValueError:  # JSON解析失败
                pos = match_pos + 1  # 后移一个字符继续尝试

        # 循环结束仍未找到有效JSON
        if last_exception:
            print(f"最后校验失败原因: {str(last_exception)}")
        return {}  # 返回空字典
