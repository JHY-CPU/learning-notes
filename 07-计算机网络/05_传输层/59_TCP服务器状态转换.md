# 60_TCP服务器状态转换

## 核心概念

- **服务器（Server）**：被动等待连接和被动关闭连接的一方
- **服务器正常路径**：CLOSED → LISTEN → SYN_RCVD → ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
- **服务器被动接受三次握手和四次挥手**

### 服务器状态转换详解

**连接建立**：
```
CLOSED
  │ 调用listen()
  ↓
LISTEN（等待SYN）
  │ 收到SYN，发送SYN+ACK
  ↓
SYN_RCVD（等待ACK）
  │ 收到ACK
  ↓
ESTABLISHED（连接已建立）
```

**连接释放**：
```
ESTABLISHED
  │ 收到FIN，发送ACK
  ↓
CLOSE_WAIT（等待自己发送FIN）
  │ 发送FIN
  ↓
LAST_ACK（等待最后的ACK）
  │ 收到ACK
  ↓
CLOSED
```

## 原理分析

### LISTEN状态

- 服务器调用listen()后进入
- 等待客户端的SYN报文段
- 监听在熟知端口上（如80、21）

### CLOSE_WAIT状态

- 服务器收到客户端的FIN后进入
- 此时服务器可能还有数据要发送
- 这就是四次挥手②和③不能合并的原因（除非服务器恰好没有数据要发了）
- 服务器在这个状态可以继续发送数据

### LAST_ACK状态

- 服务器发送FIN后进入
- 等待客户端的最后一个ACK
- 收到ACK后进入CLOSED

### 半连接队列（SYN_RCVD状态）

- 大量SYN_RCVD状态的连接 = **SYN Flood攻击**的表现
- 服务器在SYN_RCVD状态为每个连接分配资源
- 如果攻击者发送大量SYN但不完成握手，资源耗尽

### SYN Flood攻击详解
- 攻击者发送大量SYN报文，但不完成三次握手
- 服务器为每个SYN_RCVD连接分配资源（半连接队列）
- 半连接队列被耗尽，合法连接无法建立
- **防御方法**：
  - SYN Cookie：不分配资源，将状态编码在SYN+ACK的序号中
  - 增大半连接队列
  - 缩短SYN+ACK重传时间

### CLOSE_WAIT的意义
- 服务器收到FIN后，进入CLOSE_WAIT
- 此时服务器仍可发送数据（半关闭状态）
- 这就是四次挥手不能合并为三次的原因
- 只有当服务器也没有数据要发时，才发送FIN

### 服务器状态变化事件表
| 当前状态 | 收到事件 | 下一状态 |
|----------|----------|----------|
| CLOSED | listen() | LISTEN |
| LISTEN | SYN | SYN_RCVD |
| SYN_RCVD | ACK | ESTABLISHED |
| ESTABLISHED | FIN | CLOSE_WAIT |
| CLOSE_WAIT | close() | LAST_ACK |
| LAST_ACK | ACK | CLOSED |

### 服务器的被动特性
- 服务器在LISTEN状态等待连接请求
- 服务器通常不主动关闭连接（避免TIME_WAIT）
- 服务器需要处理大量并发连接
- 半连接队列和全连接队列是服务器性能关键

### 408常考要点
- 服务器被动打开和被动关闭
- LISTEN → SYN_RCVD → ESTABLISHED（三次握手服务器侧）
- ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED（四次挥手服务器侧）
- SYN Flood攻击表现为大量SYN_RCVD状态
- CLOSE_WAIT状态允许服务器继续发送数据
- 四次挥手不能合并因为CLOSE_WAIT期间可能有数据发送

## 直观理解

- **服务器像接电话的人**：你在家等着电话响（LISTEN），有人打来你接听（SYN_RCVD），通了（ESTABLISHED），对方说完了（收到FIN），你说好的然后自己也说完（CLOSE_WAIT→LAST_ACK），挂断（CLOSED）
- **SYN Flood像"骚扰电话"**：打很多电话但不说完，让接线员忙不过来

## 协议关联

- **与客户端状态转换**：互为镜像
- **与SYN Flood攻击**：大量SYN_RCVD状态
- **408常见考法**：服务器在某个事件后处于什么状态
- **SYN Cookie是防御SYN Flood的重要技术
