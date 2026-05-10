# Python requests库


## 🌐 Python requests 库


GET/POST 请求、会话 Session、超时与重试、SSL 验证、文件上传下载、代理、Cookie、Session 保持。


## GET 请求


```
// ========== requests.get ==========
# pip install requests

import requests

# 最简单的 GET 请求
response = requests.get("https://httpbin.org/get")
print(response.status_code)   # 200
print(response.text)          # 响应体 (字符串)
print(response.content)       # 响应体 (字节)
print(response.headers)       # 响应头
print(response.url)           # 最终 URL (跟随重定向后)

# 带查询参数
params = {"page": 1, "limit": 20, "sort": "name"}
response = requests.get("https://httpbin.org/get", params=params)
print(response.url)
# https://httpbin.org/get?page=1&limit=20&sort=name

# JSON 响应
data = response.json()  # 自动解析 JSON
print(data["args"])

# 自定义请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0)",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Authorization": "Bearer token123",
}
response = requests.get("https://httpbin.org/headers", headers=headers)
print(response.json())
```


## POST 与其它方法


```
// ========== POST 请求 ==========
import requests

# JSON 请求体
payload = {"name": "Alice", "email": "alice@example.com"}
response = requests.post("https://httpbin.org/post", json=payload)
print(response.json()["json"])  # {'name': 'Alice', 'email': ...}

# 表单数据
form_data = {"username": "alice", "password": "secret123"}
response = requests.post("https://httpbin.org/post", data=form_data)
print(response.json()["form"])

# 多部分表单 (文件上传)
files = {
    "file": ("report.pdf", open("report.pdf", "rb"), "application/pdf"),
    "image": ("photo.jpg", open("photo.jpg", "rb"), "image/jpeg"),
}
response = requests.post("https://httpbin.org/post", files=files)
print(response.json()["files"])

# PUT / PATCH / DELETE
response = requests.put("https://httpbin.org/put", json={"key": "value"})
response = requests.patch("https://httpbin.org/patch", json={"field": "new"})
response = requests.delete("https://httpbin.org/delete")

# HEAD / OPTIONS
response = requests.head("https://httpbin.org/get")  # 只有头,无响应体
print(response.headers)

response = requests.options("https://httpbin.org/get")  # 查看支持的 HTTP 方法
print(response.headers.get("allow"))
```


## 会话 Session


```
// ========== Session ==========
import requests

# Session: 自动保持 Cookie,复用连接
# 比多次 requests.get() 更高效

session = requests.Session()

# 设置默认头 (所有请求自动携带)
session.headers.update({
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json",
})

# 设置默认参数
session.params.update({"api_key": "abc123"})

# 登录 (Cookie 自动保存)
login_data = {"username": "admin", "password": "secret"}
session.post("https://httpbin.org/post", data=login_data)

# 后续请求自动携带登录 Cookie
response = session.get("https://httpbin.org/cookies")
print(response.json())

# 关闭会话
session.close()

# 使用上下文管理器 (推荐)
with requests.Session() as s:
    s.headers.update({"Authorization": "Bearer token"})
    resp1 = s.get("https://httpbin.org/get")
    resp2 = s.post("https://httpbin.org/post", json={"data": "value"})
    # 自动关闭

# ========== 从响应中获取 Cookie ==========
response = requests.get("https://httpbin.org/cookies/set?name=value")
for cookie in response.cookies:
    print(f"{cookie.name} = {cookie.value}")

# 手动设置 Cookie
cookies = {"session_id": "abc123", "theme": "dark"}
response = requests.get("https://httpbin.org/cookies", cookies=cookies)
```


## 超时与重试


```
// ========== 超时 ==========
import requests

# timeout: 等待服务器响应的秒数
# (连接超时, 读取超时)
try:
    response = requests.get(
        "https://httpbin.org/delay/5",
        timeout=(3, 10)  # (connect_timeout, read_timeout)
    )
except requests.Timeout:
    print("请求超时!")

# 单一超时值 (连接和读取都用这个值)
response = requests.get("https://httpbin.org/delay/1", timeout=5)

# 永不超时 (不推荐)
# response = requests.get("https://example.com", timeout=None)

# ========== 重试机制 ==========
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

# 配置重试策略:
retry_strategy = Retry(
    total=3,                # 总重试次数
    backoff_factor=1,       # 退避因子: 1, 2, 4, 8... 秒
    status_forcelist=[500, 502, 503, 504],  # 这些状态码触发重试
    allowed_methods=["GET", "POST"],  # 允许重试的方法
    raise_on_status=False,  # 重试耗尽后不抛出异常
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 此时所有请求都有重试机制
response = session.get("https://httpbin.org/status/503")
print(f"最终状态: {response.status_code}")
```


## SSL 与代理


```
// ========== SSL 验证 ==========
import requests

# verify: SSL 证书验证 (默认 True)

# 跳过验证 (仅测试环境)
response = requests.get("https://self-signed.badssl.com", verify=False)
# 会提示 InsecureRequestWarning

# 使用自定义 CA 证书
response = requests.get("https://example.com", verify="/path/to/cert.pem")

# 客户端证书
response = requests.get(
    "https://example.com",
    cert=("/path/to/client.crt", "/path/to/client.key")
)

# ========== 代理 ==========
proxies = {
    "http": "http://10.10.1.10:3128",
    "https": "http://10.10.1.10:1080",
}

# 使用代理
response = requests.get("https://httpbin.org/get", proxies=proxies)

# 带认证的代理
proxies = {
    "http": "http://user:pass@10.10.1.10:3128",
}

# 环境变量方式: $export HTTP_PROXY=http://proxy:8080

# ========== 流式响应 ==========
# 大文件下载,不加载到内存
response = requests.get("https://httpbin.org/image/png", stream=True)
with open("image.png", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# ========== 原始响应 ==========
response = requests.get("https://httpbin.org/get", stream=True)
print(response.raw.read(100))  # 读取原始 socket 数据
```


## 响应处理与错误


```
// ========== 响应对象 ==========
import requests

response = requests.get("https://httpbin.org/get")

# 状态码
print(response.status_code)           # 200
print(response.ok)                    # True (status_code < 400)
print(response.reason)                # "OK"

# 编码
print(response.encoding)              # 'utf-8'
response.encoding = 'utf-8'           # 手动设置编码
print(response.apparent_encoding)     # 自动检测编码

# 内容
print(response.text)                  # 字符串 (自动解码)
print(response.content)               # 字节
print(response.json())                # JSON 解析

# 头信息
print(response.headers)
print(response.headers['Content-Type'])
print(response.headers.get('Set-Cookie'))

# 历史重定向
print(response.history)               # 重定向历史列表
print(response.url)                   # 最终 URL
print(response.elapsed)               # 耗时 timedelta

# ========== 异常处理 ==========
try:
    response = requests.get("https://httpbin.org/status/404")
    response.raise_for_status()  # 4xx/5xx 时抛出 HTTPError
except requests.HTTPError as e:
    print(f"HTTP 错误: {e}")
except requests.ConnectionError as e:
    print(f"连接错误: {e}")
except requests.Timeout as e:
    print(f"超时: {e}")
except requests.RequestException as e:
    print(f"请求异常: {e}")

# raise_for_status 常用模式:
def fetch(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()  # 简单直接
    return resp.json()
```


> **Note:** 💡 requests 是最流行的 Python HTTP 库。Session 复用连接和 Cookie,推荐使用。配置重试和超时避免请求卡死。stream=True 处理大响应不占内存。


## 完整示例: 带重试的健壮客户端


```
// ========== 健壮 HTTP 客户端 ==========
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

class RobustClient:
    """带重试和超时的 HTTP 客户端"""

    def __init__(self, retries=3, backoff=0.5, timeout=30):
        self.timeout = timeout
        self.session = requests.Session()

        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def post(self, url, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.post(url, **kwargs)
        resp.raise_for_status()
        return resp

    def close(self):
        self.session.close()

# 使用
client = RobustClient(retries=3, timeout=10)
try:
    data = client.get("https://httpbin.org/get").json()
    print(data)
finally:
    client.close()
```


## 练习


<!-- Converted from: 111_Python requests库.html -->
