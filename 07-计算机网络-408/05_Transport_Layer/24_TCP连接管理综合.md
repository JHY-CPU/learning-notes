# 24_TCP连接管理综合

## 核心概念

- **TCP连接管理 = 连接建立（三次握手）+ 数据传输 + 连接释放（四次挥手）**
- **连接建立**：同步ISN、协商参数、分配资源
- **连接释放**：双方分别关闭，TIME_WAIT等待2MSL
- **异常处理**：RST报文段用于异常终止连接

### TCP连接管理完整流程

```
三次握手                     数据传输                    四次挥手
───────────                ──────────                ──────────
SYN →                      双向数据传输               FIN →
← SYN+ACK                  (全双工)                   ← ACK
ACK →                                                    ← FIN
ESTABLISHED                                            ACK →
                                                       TIME_WAIT(2MSL)
                                                       CLOSED
```

## 原理分析

### 同时打开（Simultaneous Open）

两个客户端同时向对方发送SYN：
```
A → SYN → B
A ← SYN ← B
双方都收到对方的SYN，各回复SYN+ACK
A ← SYN+ACK ← B
A → SYN+ACK → B
```
这种情况下四条报文就建立了连接，双方都经过 SYN_SENT → SYN_RCVD → ESTABLISHED。

### 同时关闭（Simultaneous Close）

双方同时发送FIN：
```
A → FIN → B
A ← FIN ← B
双方各回复ACK
A → ACK → B
A ← ACK ← B
双方都进入TIME_WAIT → CLOSED
```

### RST报文段

RST（Reset）用于异常情况：
1. 连接到一个不存在的端口
2. 异常终止一个连接
3. 收到一个不属于任何连接的报文段

## 直观理解

- **连接管理就像一个完整的会话**：先打招呼建立联系，交流完毕后分别说再见，最后确认挂断

## 协议关联

- **与安全**：SYN Flood攻击利用三次握手、TCP序列号猜测攻击利用ISN可预测性
- **与应用层**：HTTP/1.1的Keep-Alive复用TCP连接，减少握手挥手开销
- **408综合考法**：
  - 结合状态转换图考全过程
  - 计算各报文段的SEQ和ACK值
  - 分析异常情况（报文丢失、同时打开/关闭）
