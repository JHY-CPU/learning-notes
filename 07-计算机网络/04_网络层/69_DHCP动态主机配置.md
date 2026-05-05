# 33_DHCP动态主机配置

## 核心概念

- **DHCP（Dynamic Host Configuration Protocol）**：动态主机配置协议
  - 基于**UDP**，客户端端口68，服务器端口67
  - 基于**C/S架构**（客户端/服务器）
  - 自动分配IP地址、子网掩码、默认网关、DNS服务器等

- **DHCP分配的信息**：
  - IP地址（含租约时间）
  - 子网掩码
  - 默认网关
  - DNS服务器地址
  - 其他（域名、NTP服务器等）

- **DHCP租约（Lease）**：
  - IP地址不是永久分配，有租约期限
  - 租约到期前需续约（通常50%租期时自动续约）
  - 租约到期后IP地址可以分配给其他主机

- **408考点**：
  - DHCP基于UDP
  - DHCP支持跨网段（需要DHCP中继代理）
  - DHCP的四步交互过程

## 原理分析

### DHCP四步交互（DORA）

```
步骤1: DHCPDISCOVER（发现）
  客户端广播，源IP=0.0.0.0，目的IP=255.255.255.255
  "有没有DHCP服务器？我需要IP地址！"

步骤2: DHCPOFFER（提供）
  服务器单播/广播回复
  "我给你提供IP地址a.b.c.d，租约时间T"

步骤3: DHCPREQUEST（请求）
  客户端广播（告知所有服务器自己的选择）
  "我选择使用服务器X提供的IP地址a.b.c.d"

步骤4: DHCPACK（确认）
  服务器确认
  "确认！IP a.b.c.d分配给你，租约开始"
```

### DHCP中继代理（Relay Agent）

- 当DHCP服务器不在同一网段时，需要DHCP中继
- 路由器作为DHCP中继代理，转发DHCP报文
- 将广播转为单播发给DHCP服务器
- 解决广播不能跨网段的问题

### DHCP续约

```
租约时间 T：
  0.5T  → 客户端发送DHCPREQUEST续约（单播）
  0.875T → 若续约失败，广播DHCPDISCOVER重新申请
  T     → 租约到期，释放IP
```

## 直观理解

**类比**：DHCP像酒店入住
- 你走进酒店（接入网络）
- 前台说："我给你301房间，住3天"（DHCPOFFER）
- 你说："好的，我要301"（DHCPREQUEST）
- 前台确认："301归你了，3天后到期"（DHCPACK）
- 住了一半时间，你去续房（续约）
- 到期没续就退房（释放IP）

**记忆口诀**：
- "Discover找，Offer给，Request要，ACK确认"
- "DORA四步走，UDP来承载"

## 协议关联

### DHCP与ARP的区别

| 特性 | DHCP | ARP |
|------|------|-----|
| 功能 | 分配IP地址 | 解析MAC地址 |
| 协议层 | 应用层（基于UDP） | 网络层 |
| 工作方式 | C/S | 广播/单播 |
| 跨网段 | 支持（中继代理） | 不支持 |

### DHCP与其他协议
- DHCP报文封装在UDP中，UDP封装在IP中
- DHCPDISCOVER使用受限广播（255.255.255.255）
- 跨网段时需要DHCP中继代理（路由器配置ip helper-address）
- DHCP取代了RARP和BOOTP

### 考试陷阱
- DHCP基于UDP，不是TCP
- DHCPDISCOVER是广播，DHCPOFFER可以是单播或广播
- DHCP支持跨网段（通过中继代理），ARP不支持
