# 59_TCP客户端状态转换

## 核心概念

- **客户端（Client）**：主动发起连接和主动关闭连接的一方
- **客户端正常路径**：CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
- **客户端发起三次握手和四次挥手**

### 客户端状态转换详解

**连接建立**：
```
CLOSED
  │ 发送SYN
  ↓
SYN_SENT（等待SYN+ACK）
  │ 收到SYN+ACK，发送ACK
  ↓
ESTABLISHED（连接已建立）
```

**连接释放**：
```
ESTABLISHED
  │ 发送FIN
  ↓
FIN_WAIT_1（等待ACK或FIN）
  │ 收到ACK（但还没收到FIN）
  ↓
FIN_WAIT_2（等待对方FIN）
  │ 收到FIN，发送ACK
  ↓
TIME_WAIT（等待2MSL）
  │ 2MSL超时
  ↓
CLOSED
```

## 原理分析

### FIN_WAIT_1的两种去向

1. 收到ACK → FIN_WAIT_2（正常情况）
2. 收到FIN → CLOSING（同时关闭）
3. 收到FIN+ACK → TIME_WAIT（同时关闭，对方合并了②③）

### FIN_WAIT_2的特殊性

- 在FIN_WAIT_2状态，客户端仍可接收数据
- 这就是"半关闭"状态——客户端不再发送，但还能接收
- 直到收到对方的FIN，才进入TIME_WAIT

### TIME_WAIT的等待

- 等待2MSL
- 如果期间收到重传的FIN，重新发送ACK并重置计时器
- 2MSL后自动进入CLOSED

## 直观理解

- **客户端像打电话的人**：你先拨号（SYN_SENT），接通了（ESTABLISHED），你说完了先挂（FIN_WAIT_1→FIN_WAIT_2），等对方也说完（收到FIN），确认挂断（TIME_WAIT），等一下确认没遗漏（2MSL），彻底挂断（CLOSED）

## 协议关联

- **与服务器状态转换**：客户端的主动打开对应服务器的被动打开
- **与三次握手/四次挥手**：状态转换图就是握手挥手的展开
- **408常见考法**：给定事件序列判断客户端处于什么状态
