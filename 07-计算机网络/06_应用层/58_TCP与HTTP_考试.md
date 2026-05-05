# 59_TCP与HTTP_考试

## 核心概念

- **408常考题型**：选择题为主，涉及TCP与HTTP的关系
- **关键考点**：
  - HTTP基于TCP
  - TCP三次握手增加1 RTT延迟
  - 持久连接复用TCP减少握手
  - HTTP/2多路复用但受TCP队头阻塞影响
- **易混淆点**：
  - HTTP本身无状态，但TCP是有连接的
  - HTTP/2用TCP，HTTP/3用UDP

## 原理分析

### 典型考题1：TCP握手延迟

**题目**：HTTP/1.1首次访问Web页面，TCP握手和HTTP请求响应各需要1 RTT，总延迟是（  ）
A. 1 RTT
B. 2 RTT
C. 3 RTT
D. 4 RTT

**答案**：B

**解析**：
- TCP握手：1 RTT
- HTTP请求响应：1 RTT
- 总计：2 RTT

### 典型考题2：持久连接

**题目**：HTTP/1.1持久连接复用TCP连接的好处是（  ）
A. 减少TCP握手次数
B. 增加安全性
C. 减少DNS查询
D. 增加带宽

**答案**：A

**解析**：
- 持久连接复用TCP，避免每次请求都握手
- 减少TCP握手次数，降低延迟

### 典型考题3：HTTP/2与TCP

**题目**：HTTP/2多路复用（  ）
A. 完全解决了队头阻塞
B. 解决了HTTP层队头阻塞
C. 没有解决队头阻塞
D. 使用UDP协议

**答案**：B

**解析**：
- HTTP/2多路复用解决HTTP层队头阻塞
- TCP层队头阻塞仍存在
- HTTP/2仍使用TCP

## 直观理解

**做题技巧**：
- HTTP基于TCP，首次延迟2 RTT
- 持久连接减少TCP握手次数
- HTTP/2解决HTTP层阻塞，TCP层仍阻塞
- HTTP/3用UDP（QUIC）

## 协议关联

- **TCP与HTTP**：HTTP基于TCP
- **TCP与HTTP/2**：HTTP/2用TCP但多路复用
- **TCP与HTTP/3**：HTTP/3用QUIC（UDP）
- **408常见组合**：TCP握手 + HTTP请求 = RTT计算题
