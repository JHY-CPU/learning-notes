# 29_HTTP请求与响应报文

## 核心概念

- **HTTP报文格式**：请求报文和响应报文都由**起始行、头部、空行、正文**四部分组成
- **请求报文**：客户端发送给服务器的报文
  - 起始行 = 方法 + URL + 版本
- **响应报文**：服务器返回给客户端的报文
  - 起始行 = 版本 + 状态码 + 原因短语
- **408考试重点**：HTTP报文格式、请求/响应报文的结构

## 原理分析

### HTTP请求报文格式

```
方法 URL 版本\r\n
头部字段名: 值\r\n
头部字段名: 值\r\n
\r\n
[正文]
```

**示例**：
```
GET /index.html HTTP/1.1\r\n
Host: www.example.com\r\n
User-Agent: Mozilla/5.0\r\n
Accept: text/html\r\n
Connection: keep-alive\r\n
\r\n
```

### HTTP响应报文格式

```
版本 状态码 原因短语\r\n
头部字段名: 值\r\n
头部字段名: 值\r\n
\r\n
[正文]
```

**示例**：
```
HTTP/1.1 200 OK\r\n
Content-Type: text/html\r\n
Content-Length: 1024\r\n
Connection: keep-alive\r\n
\r\n
<html>...</html>
```

### HTTP头部字段

**通用头部**（请求和响应都可用）：
- `Connection: keep-alive / close`
- `Date:` 报文创建时间
- `Cache-Control:` 缓存控制

**请求头部**：
- `Host:` 目标主机（HTTP/1.1必需）
- `User-Agent:` 客户端信息
- `Accept:` 可接受的内容类型
- `Accept-Language:` 可接受的语言
- `Cookie:` 状态信息

**响应头部**：
- `Server:` 服务器信息
- `Content-Type:` 内容类型
- `Content-Length:` 内容长度
- `Set-Cookie:` 设置Cookie

**实体头部**（描述正文）：
- `Content-Type:` 内容类型
- `Content-Length:` 内容长度
- `Content-Encoding:` 内容编码

### HTTP报文的CRLF

- HTTP报文使用**CRLF（\r\n）**作为行结束符
- 头部与正文之间用**空行（\r\n\r\n）**分隔
- 这是408的常见考点

## 直观理解

**HTTP报文就像信件**：
- **请求报文** = "你写给商家的订单信"
  - 起始行 = "我要买什么（方法+URL+版本）"
  - 头部 = "我的联系方式和要求"
  - 正文 = "附加说明（如POST的数据）"

- **响应报文** = "商家的回信"
  - 起始行 = "订单状态（版本+状态码+原因）"
  - 头部 = "包裹信息（大小、类型）"
  - 正文 = "实际商品（HTML、图片等）"

**记忆技巧**：
- 请求起始行 = 方法 + URL + 版本
- 响应起始行 = 版本 + 状态码 + 原因短语
- 头部与正文用空行分隔
- 行结束用CRLF（\r\n）

## 协议关联

- **HTTP与TCP**：HTTP报文通过TCP传输
- **HTTP与MIME**：HTTP使用MIME格式的Content-Type
- **HTTP与Cookie**：Cookie通过HTTP头部传递
- **408考点**：
  - HTTP报文格式
  - 请求报文和响应报文的区别
  - CRLF的作用
- **陷阱**：HTTP/1.1的Host头部是必需的
