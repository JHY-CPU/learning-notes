# 61_TCP状态转换_概念_考试

## 核心概念

- **TCP状态转换图是408重点**
- **客户端路径**：CLOSED→SYN_SENT→ESTABLISHED→FIN_WAIT_1→FIN_WAIT_2→TIME_WAIT→CLOSED
- **服务器路径**：CLOSED→LISTEN→SYN_RCVD→ESTABLISHED→CLOSE_WAIT→LAST_ACK→CLOSED
- **11个状态必须记住**

### 408高频考点

**考点1：特定事件后的状态**

| 当前状态 | 事件 | 下一状态 |
|---------|------|---------|
| CLOSED | 发送SYN | SYN_SENT |
| LISTEN | 收到SYN | SYN_RCVD |
| SYN_SENT | 收到SYN+ACK | ESTABLISHED |
| SYN_RCVD | 收到ACK | ESTABLISHED |
| ESTABLISHED | 发送FIN | FIN_WAIT_1 |
| FIN_WAIT_1 | 收到ACK | FIN_WAIT_2 |
| FIN_WAIT_1 | 收到FIN | CLOSING |
| FIN_WAIT_2 | 收到FIN | TIME_WAIT |
| CLOSE_WAIT | 发送FIN | LAST_ACK |
| LAST_ACK | 收到ACK | CLOSED |
| TIME_WAIT | 2MSL超时 | CLOSED |

**考点2：CLOSE_WAIT状态的含义**

CLOSE_WAIT表示：我收到了对方的FIN（对方要关闭），我回复了ACK，但我自己还有数据要发。

## 原理分析

### 典型真题

**题目**：TCP连接中，服务器收到FIN后首先处于什么状态？

**解析**：**CLOSE_WAIT**状态。服务器回复ACK后进入CLOSE_WAIT，等待自己发送FIN。

**题目**：客户端发送FIN后处于什么状态？收到对方的FIN+ACK后呢？

**解析**：
- 发送FIN后：**FIN_WAIT_1**
- 收到FIN+ACK后：**TIME_WAIT**（跳过了FIN_WAIT_2，因为对方合并了②③）

**题目**：TIME_WAIT状态位于哪一方？持续多长时间？

**解析**：位于**主动关闭方**（通常是客户端），持续**2MSL**。

**题目**：如果服务器在CLOSE_WAIT状态收到了客户端发来的数据，应该如何处理？

**解析**：正常接收。CLOSE_WAIT只表示服务器不再发送FIN（还未发送），但仍可接收数据。CLOSE_WAIT状态的服务器到客户端方向仍然是通的。

## 直观理解

- **记忆口诀**：客户端"三次握手SYN_SENT等，四次挥手FIN_WAIT走"；服务器"三次握手LISTEN等SYN，四次挥手CLOSE_WAIT收尾"

## 协议关联

- **与socket API**：connect()→SYN_SENT, accept()→SYN_RCVD, close()→FIN_WAIT_1
- **408常见陷阱**：
  - "TIME_WAIT是服务器状态"——错误，是主动关闭方状态
  - "服务器一定是被动关闭"——不一定是，服务器也可以主动关闭
