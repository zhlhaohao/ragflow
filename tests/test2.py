import re

def convert_mixed_utf_string(input_str):
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

# 示例用法
if __name__ == "__main__":
    # 示例 1: 包含转义字符的字符串
    mixed_str1 = "Hello \\u4e16\\u754c! 你好世界!"
    result1 = convert_mixed_utf_string(mixed_str1)
    print(f"示例 1 输入: {mixed_str1}")
    print(f"示例 1 输出: {result1}")
    
    # 示例 2: 包含 UTF-8 和转义字符的字符串
    mixed_str2 = "Python \\u7f16\\u7a0b is fun! 编程真有趣!"
    result2 = convert_mixed_utf_string(mixed_str2)
    print(f"示例 2 输入: {mixed_str2}")
    print(f"示例 2 输出: {result2}")