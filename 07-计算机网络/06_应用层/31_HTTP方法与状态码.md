# 32_HTTP方法与状态码

## 核心概念

- **HTTP方法**：定义对资源的操作类型
  - **GET**：获取资源（最常用）
  - **POST**：提交数据（如表单）
  - **PUT**：更新资源（完整替换）
  - **DELETE**：删除资源
  - **HEAD**：获取头部（不获取正文）
  - **OPTIONS**：查询支持的方法
  - **TRACE**：追踪路径（诊断）
  - **PATCH**：部分更新资源
- **HTTP状态码**：表示请求处理结果
  - 1xx：信息响应
  - 2xx：成功
  - 3xx：重定向
  - 4xx：客户端错误
  - 5xx：服务器错误
- **408考试重点**：常用方法（GET/POST/PUT/DELETE）、常见状态码（200/301/302/404/500）

## 原理分析

### HTTP方法详解

| 方法 | 安全 | 幂等 | 含义 | 请求体 |
|------|------|------|------|--------|
| GET | 是 | 是 | 获取资源 | 无 |
| POST | 否 | 否 | 提交数据 | 有 |
| PUT | 否 | 是 | 更新资源 | 有 |
| DELETE | 否 | 是 | 删除资源 | 无 |
| HEAD | 是 | 是 | 获取头部 | 无 |
| OPTIONS | 是 | 是 | 查询方法 | 无 |

- **安全方法**：不修改服务器资源（GET、HEAD、OPTIONS）
- **幂等方法**：多次执行结果相同（GET、PUT、DELETE、HEAD）

### HTTP状态码详解

#### 2xx 成功
- **200 OK**：请求成功
- **201 Created**：资源创建成功
- **204 No Content**：成功但无内容

#### 3xx 重定向
- **301 Moved Permanently**：永久重定向
  - 浏览器会缓存，以后直接访问新地址
  - `Location: http://new-url.com`
- **302 Found**：临时重定向
  - 浏览器不缓存，下次还访问原地址
  - `Location: http://temp-url.com`
- **304 Not Modified**：资源未修改（缓存相关）

#### 4xx 客户端错误
- **400 Bad Request**：请求语法错误
- **401 Unauthorized**：需要认证
- **403 Forbidden**：禁止访问
- **404 Not Found**：资源不存在
- **405 Method Not Allowed**：方法不允许

#### 5xx 服务器错误
- **500 Internal Server Error**：服务器内部错误
- **502 Bad Gateway**：网关错误
- **503 Service Unavailable**：服务不可用
- **504 Gateway Timeout**：网关超时

### GET vs POST区别

| 特性 | GET | POST |
|------|-----|------|
| 参数位置 | URL查询字符串 | 请求体 |
| 参数长度 | 受URL长度限制 | 无限制 |
| 安全性 | 参数可见，不安全 | 相对安全 |
| 幂等性 | 幂等 | 非幂等 |
| 缓存 | 可缓存 | 通常不缓存 |
| 书签 | 可收藏 | 不可收藏 |

## 直观理解

**HTTP方法就像餐厅操作**：
- GET = "看看菜单"（只看不改）
- POST = "点餐"（提交新数据）
- PUT = "换一道菜"（完整替换）
- DELETE = "取消订单"（删除）
- HEAD = "只看菜单名"（不看详情）

**HTTP状态码就像餐厅反馈**：
- 200 = "好的，马上就来"（成功）
- 301 = "这家店搬到新地址了"（永久重定向）
- 302 = "今天在另一家分店营业"（临时重定向）
- 404 = "这道菜没有"（资源不存在）
- 500 = "厨房出问题了"（服务器错误）

**记忆口诀**：
- "2成功3重定向4客户端5服务器"
- "200成功301永重302临重404没有500错误"

## 代码示例

### 使用 Python requests 体验各种 HTTP 方法

```python
import requests

base_url = 'http://httpbin.org'

# GET - 获取资源（安全、幂等）
resp = requests.get(f'{base_url}/get', params={'page': 1, 'size': 10})
print(f"GET 状态码: {resp.status_code}")          # 200
print(f"请求URL: {resp.url}")                      # 包含查询参数

# POST - 提交数据（非安全、非幂等）
resp = requests.post(f'{base_url}/post', json={'name': '张三', 'age': 20})
print(f"POST 状态码: {resp.status_code}")          # 201 或 200

# PUT - 完整更新资源（非安全、幂等）
resp = requests.put(f'{base_url}/put', json={'name': '李四', 'age': 22})
print(f"PUT 状态码: {resp.status_code}")           # 200

# DELETE - 删除资源（非安全、幂等）
resp = requests.delete(f'{base_url}/delete')
print(f"DELETE 状态码: {resp.status_code}")        # 200

# HEAD - 只获取响应头部（安全、幂等）
resp = requests.head(f'{base_url}/get')
print(f"HEAD 状态码: {resp.status_code}")          # 200
print(f"Content-Length: {resp.headers.get('Content-Length')}")
print(f"响应体(为空): {resp.text}")                 # HEAD 不返回正文
```

### 常见 HTTP 状态码演示

```python
import requests

# 200 OK - 请求成功
resp = requests.get('http://httpbin.org/status/200')
print(f"200 OK: {resp.status_code}")

# 301 永久重定向
resp = requests.get('http://httpbin.org/redirect-to',
                     params={'url': 'http://example.com', 'status_code': 301},
                     allow_redirects=False)
print(f"301 Location: {resp.headers.get('Location')}")

# 302 临时重定向
resp = requests.get('http://httpbin.org/redirect-to',
                     params={'url': 'http://example.com', 'status_code': 302},
                     allow_redirects=False)
print(f"302 Location: {resp.headers.get('Location')}")

# 404 Not Found - 资源不存在
resp = requests.get('http://httpbin.org/status/404')
print(f"404 Not Found: {resp.status_code}")

# 500 Internal Server Error
resp = requests.get('http://httpbin.org/status/500')
print(f"500 Server Error: {resp.status_code}")
```

### 使用 curl 测试 HTTP 方法和状态码

```bash
# GET 请求
curl -X GET http://httpbin.org/get

# POST 请求（带 JSON 数据）
curl -X POST -H "Content-Type: application/json" \
     -d '{"name":"test"}' http://httpbin.org/post

# 查看 404 状态码
curl -I http://httpbin.org/status/404
# 输出: HTTP/1.1 404 NOT FOUND

# 跟踪 301 重定向（-L 自动跟随）
curl -v -L http://httpbin.org/redirect-to?url=http://example.com&status_code=301
```

## 协议关联

- **HTTP方法与REST**：RESTful API使用GET/POST/PUT/DELETE
- **HTTP状态码与缓存**：304与缓存机制相关
- **HTTP状态码与Cookie**：401与认证相关
- **408考点**：
  - 常用HTTP方法
  - 常见状态码的含义
  - GET与POST的区别
  - 301与302的区别
- **陷阱**：GET请求的参数在URL中，POST请求的参数在请求体中
