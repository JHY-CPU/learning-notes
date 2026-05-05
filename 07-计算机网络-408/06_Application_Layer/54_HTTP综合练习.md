# 54_HTTP综合练习

## 核心概念

- **HTTP综合练习题型**：涉及HTTP请求/响应分析、RTT计算、版本对比
- **关键考点**：
  - HTTP报文格式
  - 非持久/持久连接的RTT计算
  - HTTP方法和状态码
  - HTTP/1.0 vs HTTP/1.1 vs HTTP/2
- **408考试重点**：HTTP访问Web的完整流程、RTT计算

## 原理分析

### 综合题1：Web访问流程

**题目**：用户访问`http://www.example.com/index.html`，描述完整流程（假设DNS已缓存）。

**参考答案**：
1. 浏览器解析URL
2. TCP三次握手：建立到服务器80端口的连接（1 RTT）
3. HTTP请求：发送`GET /index.html HTTP/1.1`（1 RTT）
4. HTTP响应：服务器返回HTML文档
5. 浏览器解析HTML，发现3个图片
6. 持久连接：连续发送3个GET请求（3 RTT，非流水线）
7. 接收3个图片
8. 关闭TCP连接

**总RTT**：1（握手）+ 1（HTML）+ 3（图片）= 5 RTT

### 综合题2：非持久 vs 持久

**题目**：访问含1个HTML和4个图片的页面，分别计算HTTP/1.0（非持久）和HTTP/1.1（持久非流水线）的RTT。

**参考答案**：
- HTTP/1.0（非持久）：5 × 2 = 10 RTT
- HTTP/1.1（持久非流水线）：1 + 1 + 4 = 6 RTT

### 综合题3：HTTP报文分析

**题目**：分析以下HTTP请求报文：

```
GET /search?q=test HTTP/1.1
Host: www.google.com
User-Agent: Mozilla/5.0
Accept: text/html
Connection: keep-alive
```

**分析**：
- 方法：GET
- URL：/search?q=test
- 版本：HTTP/1.1
- Host：www.google.com（必需）
- Connection：keep-alive（持久连接）

### 综合题4：状态码判断

**题目**：以下HTTP响应表示什么？
```
HTTP/1.1 301 Moved Permanently
Location: https://www.example.com/
```

**分析**：
- 状态码301：永久重定向
- 新地址：https://www.example.com/
- 浏览器会缓存新地址

## 直观理解

**做题技巧**：
- Web访问 = DNS + TCP握手 + HTTP请求响应
- 非持久 = 每个对象2 RTT
- 持久 = 首个2 RTT，后续各1 RTT
- 流水线 = 总共3 RTT

## 协议关联

- **HTTP与TCP**：HTTP基于TCP
- **HTTP与DNS**：HTTP请求前需要DNS解析
- **HTTP与HTML**：HTTP传输HTML文档
- **408常见组合**：DNS + TCP + HTTP = Web访问综合分析
