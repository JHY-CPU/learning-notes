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

### TIME_WAIT的两个原因
1. **确保最后的ACK到达**：如果最后的ACK丢失，对方会重传FIN，TIME_WAIT期间可以重发ACK
2. **让旧连接的报文段在网络中消亡**：2MSL足够让旧连接的所有报文段从网络中消失

### TIME_WAIT过多的问题
- 服务器上大量TIME_WAIT连接占用端口资源
- **解决方案**：
  - 缩短TIME_WAIT时间（不推荐）
  - 开启SO_REUSEADDR选项
  - 让客户端主动关闭（服务器被动关闭）
  - TCP TIME_WAIT重用（tcp_tw_reuse）

### 同时打开和同时关闭
**同时打开**：双方同时发送SYN
- CLOSED → SYN_SENT → SYN_RCVD → ESTABLISHED

**同时关闭**：双方同时发送FIN
- ESTABLISHED → FIN_WAIT_1 → CLOSING → TIME_WAIT → CLOSED

### 客户端状态变化事件表
| 当前状态 | 收到事件 | 下一状态 |
|----------|----------|----------|
| CLOSED | connect() | SYN_SENT |
| SYN_SENT | SYN+ACK | ESTABLISHED |
| ESTABLISHED | close() | FIN_WAIT_1 |
| FIN_WAIT_1 | ACK | FIN_WAIT_2 |
| FIN_WAIT_1 | FIN | CLOSING |
| FIN_WAIT_2 | FIN | TIME_WAIT |
| TIME_WAIT | 2MSL超时 | CLOSED |

### 408常考要点
- 客户端主动打开和主动关闭
- TIME_WAIT等待2MSL（两个原因）
- FIN_WAIT_1可到FIN_WAIT_2、CLOSING、TIME_WAIT三种状态
- SYN和FIN各消耗1个序号
- 半关闭：FIN_WAIT_2状态可接收数据

## 直观理解

- **客户端像打电话的人**：你先拨号（SYN_SENT），接通了（ESTABLISHED），你说完了先挂（FIN_WAIT_1→FIN_WAIT_2），等对方也说完（收到FIN），确认挂断（TIME_WAIT），等一下确认没遗漏（2MSL），彻底挂断（CLOSED）
- **TIME_WAIT像"等一下再走"**——确保对方收到了你的最后回复

## 协议关联

- **与服务器状态转换**：客户端的主动打开对应服务器的被动打开
- **与三次握手/四次挥手**：状态转换图就是握手挥手的展开
- **408常见考法**：给定事件序列判断客户端处于什么状态
- **TIME_WAIT是408高频考点**
