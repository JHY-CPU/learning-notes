# 40_Ping与Traceroute

## 核心概念

- **Ping（Packet Internet Groper）**：
  - 功能：测试两台主机之间的连通性
  - 原理：发送ICMP回显请求（Type=8），接收ICMP回显应答（Type=0）
  - 使用网络层ICMP协议，不经过传输层（TCP/UDP）
  - 通过往返时间（RTT）判断网络延迟

- **Traceroute（Windows下为tracert）**：
  - 功能：跟踪数据报从源到目的经过的路由器
  - 原理：利用ICMP时间超过（Type=11）报文
  - 通过逐步增加TTL值逐跳发现路由器

- **408考点**：
  - Ping使用ICMP回显请求/应答
  - Traceroute使用ICMP时间超过和目的不可达
  - Ping和Traceroute都工作在网络层
  - Traceroute逐步增加TTL来发现路径

## 原理分析

### Ping的工作过程

```
步骤1: 主机A向主机B发送ICMP回显请求
       封装在IP数据报中：源IP=A，目的IP=B
       ICMP类型=8，代码=0

步骤2: 主机B收到后，回复ICMP回显应答
       封装在IP数据报中：源IP=B，目的IP=A
       ICMP类型=0，代码=0

步骤3: 主机A收到应答，计算RTT
       RTT = 收到时间 - 发送时间
```

### Traceroute的工作过程

```
步骤1: 发送TTL=1的UDP数据报（目的端口=很大的数，如33434）
       → 第一个路由器将TTL减为0，丢弃，返回ICMP时间超过
       → 记录第一跳路由器IP和RTT

步骤2: 发送TTL=2的UDP数据报
       → 第一个路由器TTL减为1，转发
       → 第二个路由器TTL减为0，丢弃，返回ICMP时间超过
       → 记录第二跳路由器IP和RTT

步骤3: 继续递增TTL...

步骤N: 发送TTL=N的UDP数据报到达目的主机
       → 目的主机收到后，端口不可达（因为UDP端口是随机大端口）
       → 返回ICMP目的不可达（Type=3, Code=3）
       → Traceroute结束
```

### Traceroute的两种实现

| 版本 | 发送的报文 | 结束条件 |
|------|-----------|---------|
| Unix/Linux | UDP（目的端口递增） | ICMP目的不可达（端口不可达） |
| Windows | ICMP回显请求 | ICMP回显应答 |

## 直观理解

**类比**：
- **Ping** = 打电话确认对方是否能接通
  - "嘟...（请求）" → "喂？（应答）" → 确认连通

- **Traceroute** = 追踪快递路线
  - 第一封信写"只能经过1个邮局" → 第一个邮局退回（告诉你它的地址）
  - 第二封信写"只能经过2个邮局" → 第二个邮局退回
  - 直到信送到收件人手中

**记忆口诀**：
- "Ping用8和0，一问一答测连通"
- "Traceroute加TTL，逐跳发现路由器"

## 协议关联

### Ping与网络层
- Ping直接使用ICMP，不经过传输层
- Ping不通不代表一定不可达（可能防火墙禁止ICMP）

### Traceroute与网络层
- 使用UDP（Unix）或ICMP（Windows）+ ICMP差错报文
- 发现的路径可能不是实际转发路径（因为负载均衡）
- 每跳发3个包（取3次RTT的平均值）

### 考试要点
- Ping使用ICMP回显请求(Type=8)/应答(Type=0)
- Traceroute使用ICMP时间超过(Type=11)和目的不可达(Type=3)
- Ping和Traceroute都不使用TCP或UDP（Linux Traceroute除外，首包用UDP）
- Ping测试连通性和延迟，Traceroute发现路径
