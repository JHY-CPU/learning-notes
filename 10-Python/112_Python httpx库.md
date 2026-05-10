# Python httpx库


## 🌐 Python httpx 库


httpx 同步与异步客户端、与 requests API 对比、AsyncClient 并发请求、超时/重试、httpx + asyncio 整合。


## httpx 同步客户端


```
// ========== httpx 基础 ==========
# pip install httpx

import httpx

# GET 请求 (API 与 requests 相似)
response = httpx.get("https://httpbin.org/get")
print(response.status_code)   # 200
print(response.text)          # 文本
print(response.json())        # JSON

# 带查询参数
response = httpx.get(
    "https://httpbin.org/get",
    params={"page": 1, "limit": 10}
)
print(response.url)

# POST JSON
response = httpx.post(
    "https://httpbin.org/post",
    json={"name": "Alice", "age": 30}
)

# POST 表单
response = httpx.post(
    "https://httpbin.org/post",
    data={"key": "value"}
)

# 自定义头
response = httpx.get(
    "https://httpbin.org/headers",
    headers={"Authorization": "Bearer token123"}
)

# ========== Client 会话 ==========
with httpx.Client() as client:
    # 复用连接, 保持 Cookie
    client.headers.update({"User-Agent": "MyApp/1.0"})

    resp1 = client.get("https://httpbin.org/cookies/set?name=alice")
    resp2 = client.get("https://httpbin.org/cookies")  # 有 Cookie
    print(resp2.json())
```


## httpx 异步客户端


```
// ========== AsyncClient ==========
import httpx
import asyncio

# 异步 HTTP 客户端 (基于 asyncio)
# 适合高并发场景

async def fetch_one():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://httpbin.org/get")
        return response.json()

async def main():
    data = await fetch_one()
    print(data)

asyncio.run(main())

# ========== 并发请求 ==========
async def fetch_all():
    urls = [
        f"https://httpbin.org/delay/{i}" for i in range(1, 4)
    ]

    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)

        results = []
        for resp in responses:
            results.append(resp.json())
        return results

async def main():
    start = asyncio.get_event_loop().time()
    results = await fetch_all()
    elapsed = asyncio.get_event_loop().time() - start
    print(f"耗时: {elapsed:.1f}s")  # ≈ 3s
    print(f"结果数: {len(results)}")

asyncio.run(main())
```


## requests vs httpx 对比


```
// ========== 对比 ==========
# httpx 是 requests 的现代替代品,额外支持:
# 1. 异步 (AsyncClient)
# 2. HTTP/2
# 3. 原生 ASGI 支持 (用于测试 FastAPI)

import httpx
import requests

# ===== requests =====
resp = requests.get("https://httpbin.org/get")
print(resp.status_code)
print(resp.json())

# ===== httpx 同步 =====
resp = httpx.get("https://httpbin.org/get")
print(resp.status_code)
print(resp.json())

# API 非常相似! 主要区别:
# httpx.Client() vs requests.Session()
# httpx 用 .get()/.post() 而非 response.json()

# ===== httpx HTTP/2 =====
# 需要安装: pip install h2
client = httpx.Client(http2=True)
resp = client.get("https://httpbin.org/get")
print(resp.http_version)  # "HTTP/2"

# ===== httpx ASGI 测试 =====
# 直接测试 ASGI 应用 (FastAPI)
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

app = FastAPI()

@app.get("/")
async def root():
    return {"msg": "Hello"}

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
    assert resp.json() == {"msg": "Hello"}
```


## httpx 进阶用法


```
// ========== 超时与重试 ==========
import httpx

# 超时配置
timeout = httpx.Timeout(
    30.0,           # 总超时
    connect=5.0,    # 连接超时
    read=10.0,      # 读取超时
    write=10.0,     # 写入超时
    pool=5.0        # 连接池等待超时
)

client = httpx.Client(timeout=timeout)

# ========== 传输选项 ==========
# 限制连接池
limits = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=100,
)

client = httpx.Client(limits=limits)

# ========== 代理 ==========
proxies = {
    "http://": "http://localhost:8080",
    "https://": "http://localhost:8080",
}
client = httpx.Client(proxies=proxies)

# ========== 事件钩子 ==========
def log_request(request):
    print(f"请求: {request.method} {request.url}")

def log_response(response):
    print(f"响应: {response.status_code}")

client = httpx.Client(
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    }
)

# ========== 错误处理 ==========
try:
    response = httpx.get("https://httpbin.org/status/500")
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    print(f"HTTP 错误: {e.response.status_code}")
except httpx.ConnectError as e:
    print(f"连接错误: {e}")
except httpx.TimeoutException as e:
    print(f"超时: {e}")

# ========== 流式响应 ==========
with httpx.stream("GET", "https://httpbin.org/drip?numbytes=100") as resp:
    for chunk in resp.iter_bytes():
        print(f"块: {chunk}")
```


> **Note:** 💡 httpx 可作为 requests 的现代替代品。同步 API 几乎相同,迁移成本低。AsyncClient 提供原生异步支持。支持 HTTP/2 和 ASGI 测试。


## 完整示例: 异步批量 API 客户端


```
// ========== 异步 API 客户端 ==========
import httpx
import asyncio
from typing import List, Dict

class AsyncAPIClient:
    """异步 API 客户端"""

    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": "AsyncClient/1.0"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def fetch_items(self, endpoint: str, ids: List[int]) -> List[Dict]:
        """并发获取多个资源"""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=httpx.Timeout(30.0),
        ) as client:
            tasks = [
                self._fetch_one(client, f"{endpoint}/{item_id}")
                for item_id in ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 过滤错误
            valid = []
            for r in results:
                if isinstance(r, dict):
                    valid.append(r)
                else:
                    print(f"请求失败: {r}")
            return valid

    async def _fetch_one(self, client, url: str):
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

    async def create_item(self, endpoint: str, data: Dict) -> Dict:
        """创建资源"""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
        ) as client:
            resp = await client.post(endpoint, json=data)
            resp.raise_for_status()
            return resp.json()

# 使用
async def main():
    client = AsyncAPIClient("https://jsonplaceholder.typicode.com")
    items = await client.fetch_items("posts", [1, 2, 3, 4, 5])
    for item in items:
        print(f"  #{item['id']}: {item['title'][:30]}...")

asyncio.run(main())
```


## 练习


<!-- Converted from: 112_Python httpx库.html -->
