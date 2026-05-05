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

## 直观理解

- **服务器像接电话的人**：你在家等着电话响（LISTEN），有人打来你接听（SYN_RCVD），通了（ESTABLISHED），对方说完了（收到FIN），你说好的然后自己也说完（CLOSE_WAIT→LAST_ACK），挂断（CLOSED）

## 协议关联

- **与客户端状态转换**：互为镜像
- **与SYN Flood攻击**：大量SYN_RCVD状态
- **408常见考法**：服务器在某个事件后处于什么状态
