import time
import logging
from rag.llm.chat_model import UniinChat, OpenAI_APIChat
from api import settings
test_max_token = True

logging.basicConfig(
    filename='test_llm.log',
    format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

class ContextLengthTester:


    def __init__(self, model_name: str):
        self.config = {"max_context_tokens": 256000, "chars_per_token": 1.0}
        # self.config = self.MODEL_CONFIG.get(model_name, self.MODEL_CONFIG["default"])
        if model_name == "Qwen3-14B":
          self.client = OpenAI_APIChat(model_name=model_name, key='sk-dyuyfgue64we6e7wyr', base_url='http://172.24.12.149:9853/v1')
        else:
          self.client = UniinChat("", model_name)
        self.char_ratio = 1 / self.config["chars_per_token"]

    def generate_chinese_text(self, desired_tokens: int) -> str:
        """生成指定token数量的中文测试文本"""
        base = "这是一段用于测试大模型上下文长度的标准中文文本，通过添加唯一标识符确保文本的唯一性。"
        unique_id = f"当前测试标识符：[{desired_tokens}-{int(time.time())}]。"

        # 计算目标字符数
        target_chars = int(desired_tokens * self.char_ratio)
        repeat_times = target_chars // (len(base) + len(unique_id)) + 1

        full_text = (base + unique_id) * repeat_times
        return full_text[:target_chars]

    def test_max_context(self, max_retry: int = 5, test_max_token = False) -> int:
        """二分法测试上下文长度"""
        low, high = 0, self.config["max_context_tokens"]
        best_success = 0
        attempt_count = 0

        while low <= high and attempt_count < max_retry * 10:
            mid = (low + high) // 2
            test_text = self.generate_chinese_text(mid)
            actual_chars = len(test_text)
            estimated_tokens = int(actual_chars / self.char_ratio)
            if test_max_token:
              gen_conf = {"frequency_penalty": 0.5, "max_tokens": mid, "presence_penalty": 0.4, "temperature": 0.5, "top_p": 0.5}
            else:
              gen_conf = {"frequency_penalty": 0.5, "max_tokens": 100, "presence_penalty": 0.4, "temperature": 0.5, "top_p": 0.5}

            print(f'测试区间 [{low}, {high}] 中点值={mid} 实际字符={actual_chars}')
            try:
                response, _ = self.client.chat(
                    system = "repeat user question exactly as is",
                    history = [
                        {"role": "user", "content": test_text}
                    ],
                    gen_conf = gen_conf
                )

                print(f"LLM response: {response}")
                if "ERROR" not in response and len(response)>0 and 'Token indices sequence length' not in response:
                    best_success = max(best_success, estimated_tokens)
                    low = mid + 1
                    print(f'✅ 成功处理约 {estimated_tokens} tokens')
                else:
                    high = mid - 1
                    print(f'⚠️  响应截断于 {estimated_tokens} tokens')

            except Exception as e:
                print(f'🚨 API 错误: {e}')
                if 'context_length' in str(e).lower():
                    print(f'🚫 上下文长度超出 {estimated_tokens}')
                    high = mid - 1
                else:
                    attempt_count += 1
                    time.sleep(2 ** attempt_count)
                continue

        return best_success

if __name__ == '__main__':
    settings.init_settings()
    model_name = 'Doubao-1.5-pro-256k'
    tester = ContextLengthTester(model_name)
    safe_max = tester.test_max_context(
        max_retry=5,
        test_max_token = True,
    )
    print(f'实测 {safe_max} tokens')



"""
Deepseek-r1: context_length:131063 max_tokens:16384
qwen3-32b: context_length:137602  max_tokens:16384
qwen3-235b-a22b: context_length:137602 max_tokens:16384
Doubao-1.5-pro-256k: 256000
Deepseek-r1-250528: context_length:196586  max_tokens:16384
Qwen3-14B: 32768
qwen2.5-coder-32b-instruct: 14061
unicom-70b-chat: 32768
"""