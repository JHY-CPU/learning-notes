# 57_HTTP报文分析_考试

## 核心概念

- **408常考题型**：选择题为主，涉及HTTP报文格式分析
- **关键考点**：
  - 请求起始行：方法+URL+版本
  - 响应起始行：版本+状态码+原因
  - 常见头部字段
  - CRLF分隔符
- **易混淆点**：
  - 请求和响应起始行格式不同
  - Host头部在HTTP/1.1中是必需的

## 原理分析

### 典型考题1：报文格式

**题目**：HTTP请求报文的起始行格式是（  ）
A. 版本 状态码 原因
B. 方法 URL 版本
C. URL 方法 版本
D. 版本 方法 URL

**答案**：B

### 典型考题2：头部分析

**题目**：HTTP/1.1请求中必须包含的头部是（  ）
A. User-Agent
B. Accept
C. Host
D. Cookie

**答案**：C

### 典型考题3：状态码分析

**题目**：HTTP响应`HTTP/1.1 301 Moved Permanently`表示（  ）
A. 请求成功
B. 永久重定向
C. 临时重定向
D. 资源不存在

**答案**：B

### 典型考题4：报文分隔

**题目**：HTTP报文头部与正文的分隔使用（  ）
A. 单个\r\n
B. \r\n\r\n（空行）
C. 特殊字符
D. 没有分隔符

**答案**：B

## 直观理解

**做题技巧**：
- 请求起始行 = 方法+URL+版本
- 响应起始行 = 版本+状态码+原因
- 空行（\r\n\r\n）分隔头部和正文
- Host在HTTP/1.1中必需

### 典型考题5：请求方法

**题目**：HTTP请求中，用于提交表单数据的常用方法是（  ）
A. GET
B. POST
C. HEAD
D. PUT

**答案**：B

**解析**：
- GET：获取资源，参数在URL中
- POST：提交数据，数据在请求体中
- HEAD：获取响应头部（不含正文）
- PUT：上传资源

### 典型考题6：Content-Type

**题目**：HTTP响应头部`Content-Type: text/html; charset=UTF-8`表示（  ）
A. 传输编码
B. 内容类型和字符集
C. 内容长度
D. 缓存控制

**答案**：B

**解析**：
- Content-Type指定MIME类型和字符集
- 常见类型：text/html、application/json、image/jpeg

### 典型考题7：Keep-Alive

**题目**：HTTP/1.1中`Connection: keep-alive`的作用是（  ）
A. 保持TCP连接
B. 加密数据
C. 压缩数据
D. 缓存数据

**答案**：A

**解析**：
- keep-alive表示使用持久连接
- 多个HTTP请求复用同一个TCP连接
- HTTP/1.1默认启用

### HTTP常见头部字段

| 头部 | 作用 | 示例 |
|------|------|------|
| Host | 目标主机（1.1必需） | Host: www.example.com |
| User-Agent | 客户端信息 | User-Agent: Mozilla/5.0 |
| Content-Type | 内容类型 | Content-Type: text/html |
| Content-Length | 内容长度 | Content-Length: 1024 |
| Cookie | 客户端Cookie | Cookie: sid=abc123 |
| Set-Cookie | 服务器设置Cookie | Set-Cookie: sid=abc123 |
| Connection | 连接管理 | Connection: keep-alive |
| Cache-Control | 缓存控制 | Cache-Control: max-age=3600 |
| Accept | 可接受的类型 | Accept: text/html |
| Location | 重定向目标 | Location: /new-page |

### 408常考要点
- 请求起始行 = 方法+URL+版本
- 响应起始行 = 版本+状态码+原因短语
- 头部与正文用空行（\r\n\r\n）分隔
- Host在HTTP/1.1中是必需头部
- Content-Type指定MIME类型
- Cookie/Set-Cookie实现状态管理
- Connection: keep-alive启用持久连接

## 协议关联

- **HTTP报文与TCP**：通过TCP传输
- **HTTP报文与Cookie**：通过Set-Cookie/Cookie头部
- **HTTP报文与SSL/TLS**：HTTPS中报文被加密
- **408常见组合**：报文格式 + 状态码 + 头部 = 综合分析
