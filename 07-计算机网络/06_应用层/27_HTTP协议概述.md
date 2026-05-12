# 28_HTTP协议概述

## 核心概念

- **HTTP（HyperText Transfer Protocol）**：超文本传输协议，WWW的核心协议
- HTTP使用**TCP**协议，端口号**80**
- HTTP是**C/S模式**的应用协议
- HTTP是**无状态（Stateless）**协议：服务器不保留客户端的状态信息
- HTTP是**无连接**的（HTTP/1.0）或支持**持久连接**（HTTP/1.1+）
- **408考试重点**：HTTP的基本特点、工作过程、与TCP的关系

## 原理分析

### HTTP工作过程

1. **DNS解析**：
   - 浏览器从URL中提取主机名
   - 调用DNS解析获得IP地址

2. **TCP连接**：
   - 浏览器与Web服务器建立TCP连接
   - 三次握手（SYN → SYN+ACK → ACK）

3. **HTTP请求**：
   - 浏览器发送HTTP请求报文
   - 包含：方法、URL、版本、头部、可能的正文

4. **HTTP响应**：
   - 服务器处理请求
   - 发送HTTP响应报文
   - 包含：版本、状态码、原因短语、头部、可能的正文

5. **关闭连接**：
   - HTTP/1.0：每个请求完成后关闭连接
   - HTTP/1.1：保持连接，可以复用

### HTTP版本演进

| 版本 | 特点 |
|------|------|
| HTTP/1.0 | 非持久连接，每次请求建立新连接 |
| HTTP/1.1 | 持久连接（默认），管道化，Host头部 |
| HTTP/2 | 多路复用，头部压缩，服务器推送 |
| HTTP/3 | 基于QUIC（UDP），减少延迟 |

### HTTP的基本特点

1. **无状态**：服务器不记住客户端的先前请求
2. **无连接**（HTTP/1.0）：每次请求独立建立连接
3. **支持持久连接**（HTTP/1.1+）：连接可以复用
4. **支持缓存**：通过Cache-Control、ETag等头部
5. **支持内容协商**：通过Accept头部协商内容类型
6. **支持代理**：可以通过代理服务器转发请求

## 直观理解

**HTTP就像餐厅点餐**：
- **无状态**：服务员不记得你上次点了什么（除非你给小费/用Cookie）
- **无连接**（HTTP/1.0）：每次点餐都要重新叫服务员
- **持久连接**（HTTP/1.1）：叫一次服务员可以点多次餐
- **请求/响应**：你说"来份牛排"（请求），服务员端上牛排（响应）

**记忆技巧**：
- HTTP = "超文本传输协议"，端口80
- HTTP无状态 → 需要Cookie/Session维护状态
- HTTP/1.0无连接 → HTTP/1.1持久连接
- HTTP基于TCP → 三次握手 + 可靠传输

**HTTP访问Web的完整过程**：
1. DNS解析（域名→IP）
2. TCP三次握手
3. HTTP请求
4. HTTP响应
5. TCP四次挥手（或保持连接）

## 代码示例

### 使用 Python requests 库发送 HTTP 请求

```python
import requests

# GET 请求 - 获取网页内容
response = requests.get('http://httpbin.org/get')
print(f"状态码: {response.status_code}")        # 200
print(f"HTTP版本: {response.raw.version}")       # 11 表示 HTTP/1.1
print(f"响应头 Content-Type: {response.headers['Content-Type']}")
print(f"响应体: {response.json()}")

# POST 请求 - 提交数据
data = {'username': 'student', 'password': '123456'}
response = requests.post('http://httpbin.org/post', data=data)
print(f"POST 状态码: {response.status_code}")
```

### 使用 Python http.server 搭建简易 HTTP 服务器

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class MyHandler(BaseHTTPRequestHandler):
    """自定义 HTTP 请求处理器"""

    def do_GET(self):
        """处理 GET 请求"""
        # 构造响应数据
        response_data = {
            'message': 'Hello, HTTP!',
            'path': self.path,
            'method': self.command
        }
        body = json.dumps(response_data, ensure_ascii=False).encode('utf-8')

        # 发送响应
        self.send_response(200)                          # 状态码 200
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()                               # 空行，分隔头部和正文
        self.wfile.write(body)                           # 发送正文

# 启动服务器，监听 8080 端口
server = HTTPServer(('localhost', 8080), MyHandler)
print('HTTP 服务器运行在 http://localhost:8080')
server.serve_forever()
```

### 使用 curl 命令观察 HTTP 报文

```bash
# 查看完整的请求和响应报文（-v 详细模式）
curl -v http://www.example.com

# 仅查看响应头部（-I 等同于 HEAD 方法）
curl -I http://www.example.com

# 指定请求方法和头部
curl -X POST -H "Content-Type: application/json" \
     -d '{"key":"value"}' http://httpbin.org/post
```

### 使用 socket 手动构造 HTTP 请求

```python
import socket

# 创建 TCP 套接字，手动发送 HTTP 请求
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('httpbin.org', 80))

# 手工构造 HTTP GET 请求报文
request = (
    "GET /get HTTP/1.1\r\n"           # 请求行: 方法 + URL + 版本
    "Host: httpbin.org\r\n"            # Host 头部 (HTTP/1.1 必需)
    "Connection: close\r\n"            # 关闭连接
    "\r\n"                             # 空行分隔头部和正文
)
sock.send(request.encode())

# 接收响应
response = sock.recv(4096).decode()
print(response)  # 可以看到完整的 HTTP 响应报文
sock.close()
```

## 协议关联

- **HTTP与TCP**：HTTP使用TCP的可靠传输，端口80
- **HTTP与DNS**：HTTP请求前需要DNS解析
- **HTTP与HTML**：HTTP传输HTML文档
- **HTTP与Cookie**：Cookie在HTTP头部传递，用于状态管理
- **HTTP与HTTPS**：HTTPS = HTTP + SSL/TLS，端口443
- **408考点**：
  - HTTP端口80
  - HTTP无状态
  - HTTP基于TCP
  - HTTP/1.0 vs HTTP/1.1的区别
- **陷阱**：HTTP无状态，但可以通过Cookie/Session实现状态管理
