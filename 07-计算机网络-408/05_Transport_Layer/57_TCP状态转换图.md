# 57_TCP状态转换图

## 核心概念

- **TCP状态转换图**：描述TCP连接生命周期中所有可能的状态和转换
- **11个TCP状态**：CLOSED、LISTEN、SYN_SENT、SYN_RCVD、ESTABLISHED、FIN_WAIT_1、FIN_WAIT_2、CLOSE_WAIT、CLOSING、TIME_WAIT、LAST_ACK
- **客户端和服务器共用同一个状态图**，只是走的路径不同
- **408必考**：状态转换图是408的重点和难点

### TCP 11个状态

| 状态 | 说明 |
|------|------|
| CLOSED | 初始状态，无连接 |
| LISTEN | 服务器监听，等待连接请求 |
| SYN_SENT | 客户端已发送SYN，等待SYN+ACK |
| SYN_RCVD | 服务器已收到SYN并回复SYN+ACK，等待ACK |
| ESTABLISHED | 连接建立，可以双向传输数据 |
| FIN_WAIT_1 | 主动关闭方已发送FIN，等待ACK或FIN |
| FIN_WAIT_2 | 主动关闭方收到ACK，等待对方FIN |
| CLOSE_WAIT | 被动关闭方收到FIN并回复ACK，等待自己发送FIN |
| CLOSING | 双方同时关闭，等待对方ACK |
| TIME_WAIT | 主动关闭方等待2MSL |
| LAST_ACK | 被动关闭方已发送FIN，等待最后的ACK |

## 原理分析

### 客户端状态转换路径（主动打开/关闭）

```
CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
```

### 服务器状态转换路径（被动打开/关闭）

```
CLOSED → LISTEN → SYN_RCVD → ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
```

### 状态转换触发条件

- **发送SYN**：从CLOSED到SYN_SENT
- **收到SYN**：从LISTEN到SYN_RCVD
- **收到SYN+ACK**：从SYN_SENT到ESTABLISHED
- **收到ACK**：从SYN_RCVD到ESTABLISHED
- **发送FIN**：从ESTABLISHED到FIN_WAIT_1
- **收到ACK**：从FIN_WAIT_1到FIN_WAIT_2
- **收到FIN**：从FIN_WAIT_2到TIME_WAIT（同时回复ACK）
- **2MSL超时**：从TIME_WAIT到CLOSED

## 直观理解

- **状态图像一张地图**：每个状态是一个"城市"，箭头是"道路"，报文段是"通行证"
- **记忆方法**：先记正常路径（握手→传输→挥手），再记异常路径（同时打开/关闭）

## 协议关联

- **与三次握手**：CLOSED → SYN_SENT → ESTABLISHED（客户端）；CLOSED → LISTEN → SYN_RCVD → ESTABLISHED（服务器）
- **与四次挥手**：ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
- **408常考**：给定事件判断状态转换
