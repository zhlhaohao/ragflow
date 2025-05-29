import ipaddress
from urllib.parse import urlparse

def extract_ip_from_url(base_url):
    # 解析URL
    parsed_url = urlparse(base_url)
    
    # 获取网络位置部分（即：'xxxx:9888'），不包括路径
    netloc = parsed_url.netloc
    
    # 如果包含端口号，则去掉端口号，只保留IP地址/域名部分
    if ':' in netloc:
        host = netloc.split(':')[0]
    else:
        host = netloc
    
    # 验证是否为IP地址
    try:
        if ipaddress.ip_address(host):
            return host
    except ValueError:
        # 不是IP地址，可能是域名
        return None



def is_ip_address(base_url):
    try:
        # 尝试将字符串解析为IP地址
        if ipaddress.ip_address(base_url):
            return True
    except ValueError:
        # 如果解析失败，则表示不是IP地址，可能是域名
        return False


# 示例
base_url = "http://127.0.0.1:9888/v1"
ip_address = extract_ip_from_url(base_url)

print(is_ip_address(ip_address))

