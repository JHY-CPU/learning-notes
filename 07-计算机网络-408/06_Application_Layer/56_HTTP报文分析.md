# 56_HTTP报文分析

## 核心概念

- **HTTP报文分析**：分析HTTP请求和响应报文的各个部分
- **请求报文结构**：起始行（方法+URL+版本）、头部、空行、正文
- **响应报文结构**：起始行（版本+状态码+原因）、头部、空行、正文
- **408考试重点**：HTTP报文格式、各字段含义

## 原理分析

### 请求报文分析

**示例**：
```
POST /login HTTP/1.1\r\n
Host: www.example.com\r\n
User-Agent: Mozilla/5.0\r\n
Content-Type: application/x-www-form-urlencoded\r\n
Content-Length: 29\r\n
Connection: keep-alive\r\n
\r\n
username=admin&password=123
```

**分析**：
- 方法：POST
- URL：/login
- 版本：HTTP/1.1
- Host：www.example.com
- Content-Type：表单数据格式
- Content-Length：正文29字节
- Connection：持久连接
- 正文：username=admin&password=123

### 响应报文分析

**示例**：
```
HTTP/1.1 200 OK\r\n
Content-Type: text/html; charset=utf-8\r\n
Content-Length: 1024\r\n
Set-Cookie: sessionId=abc123\r\n
Connection: keep-alive\r\n
\r\n
<html>...</html>
```

**分析**：
- 版本：HTTP/1.1
- 状态码：200
- 原因：OK
- Content-Type：HTML文档
- Content-Length：正文1024字节
- Set-Cookie：设置会话Cookie
- 正文：HTML内容

### 常见头部字段

| 字段 | 类型 | 说明 |
|------|------|------|
| Host | 请求 | 目标主机（HTTP/1.1必需） |
| User-Agent | 请求 | 客户端信息 |
| Accept | 请求 | 可接受的内容类型 |
| Cookie | 请求 | 客户端Cookie |
| Content-Type | 实体 | 内容类型 |
| Content-Length | 实体 | 内容长度 |
| Set-Cookie | 响应 | 设置Cookie |
| Location | 响应 | 重定向地址 |
| Server | 响应 | 服务器信息 |

## 直观理解

**报文就像快递单**：
- 起始行 = 快递单号和类型
- 头部 = 收发件人信息、包裹说明
- 空行 = 分隔线
- 正文 = 实际包裹内容

## 协议关联

- **HTTP报文与TCP**：HTTP报文通过TCP传输
- **HTTP报文与Cookie**：Cookie通过Set-Cookie/Cookie头部传递
- **HTTP报文与MIME**：Content-Type使用MIME格式
- **408考点**：报文格式、各字段含义、CRLF分隔
