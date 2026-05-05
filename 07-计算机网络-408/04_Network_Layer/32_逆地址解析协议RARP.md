# 32_逆地址解析协议RARP

## 核心概念

- **RARP（Reverse Address Resolution Protocol）**：逆地址解析协议
  - 功能：将MAC地址解析为IP地址
  - 与ARP相反：ARP是 IP→MAC，RARP是 MAC→IP
  - 已基本被DHCP取代

- **RARP的工作原理**：
  1. 无盘工作站（diskless workstation）启动时知道自己的MAC地址但不知道IP
  2. 广播发送RARP请求（包含自己的MAC）
  3. RARP服务器收到后查找MAC-IP映射表
  4. RARP服务器单播回复RARP应答（包含该MAC对应的IP）

- **RARP的局限性**：
  - 需要在每个网段配置RARP服务器
  - 只能获取IP地址，不能获取子网掩码、网关等信息
  - 无法跨网段工作

- **RARP的替代方案**：
  - **BOOTP**：扩展了RARP，可获取更多信息
  - **DHCP**：BOOTP的升级版，动态分配IP，完全取代RARP

## 原理分析

### RARP vs ARP

| 特性 | ARP | RARP |
|------|-----|------|
| 方向 | IP → MAC | MAC → IP |
| 发起方 | 知道目的IP的主机 | 不知道自己IP的主机 |
| 用途 | 获取下一跳MAC | 获取自身IP |
| 现状 | 广泛使用 | 已被DHCP取代 |
| 请求方式 | 广播 | 广播 |
| 应答方式 | 单播 | 单播 |

### RARP工作流程

```
步骤1: 无盘工作站启动，知道自己的MAC地址
步骤2: 广播RARP请求："我的MAC是XX:XX:XX:XX:XX:XX，请告诉我我的IP"
步骤3: RARP服务器收到请求，查找映射表
步骤4: RARP服务器单播回复："你的IP是a.b.c.d"
步骤5: 工作站获得IP地址，配置网络
```

### RARP → BOOTP → DHCP 演进

```
RARP（仅获取IP）
  ↓ 不足：不能获取子网掩码、网关等
BOOTP（获取IP+子网掩码+网关+DNS等）
  ↓ 不足：静态配置，不够灵活
DHCP（动态分配，租约机制，支持移动性）
```

## 直观理解

**类比**：
- RARP像一个人走进办公室说："我的工牌号是12345，请问我叫什么名字？"
- 办公室管理员（RARP服务器）查表后回答："你叫张三"
- 现在这个人（主机）就知道了自己的名字（IP地址）

**记忆口诀**：
- "ARP：知道名字问地址；RARP：知道地址问名字"
- "RARP老古董，DHCP已替代"

## 协议关联

### RARP与DHCP的关系

- DHCP完全取代了RARP的功能
- DHCP不仅分配IP，还提供子网掩码、默认网关、DNS服务器等
- DHCP使用UDP（端口67/68），RARP工作在数据链路层
- 现代网络中不存在RARP服务器

### 408考试注意
- RARP通常只作为选择题中的干扰项
- 了解RARP的MAC→IP功能即可
- 重点掌握其替代者DHCP
- RARP和ARP都是网络层协议
