# 44_DHCP应用层详解

## 核心概念

- **DHCP（Dynamic Host Configuration Protocol）**：动态主机配置协议
- DHCP使用**UDP**协议：
  - **服务器端口67**
  - **客户端端口68**
- DHCP是**C/S模式**的应用协议
- DHCP自动分配：
  - **IP地址**
  - **子网掩码**
  - **默认网关**
  - **DNS服务器地址**
- **408考试重点**：DHCP端口号、工作过程、与BOOTP的关系

## 原理分析

### DHCP工作过程（DORA）

1. **DHCP Discover（发现）**：
   - 客户端广播DHCP Discover报文
   - 源IP：0.0.0.0，目的IP：255.255.255.255
   - 源端口：68，目的端口：67
   - 客户端还不知道DHCP服务器在哪里

2. **DHCP Offer（提供）**：
   - DHCP服务器收到Discover后，发送Offer报文
   - 包含：IP地址、子网掩码、租期等
   - 可以是广播或单播

3. **DHCP Request（请求）**：
   - 客户端选择一个Offer，发送Request报文
   - 广播发送，告知所有服务器选择了哪个
   - 其他服务器收回未被选择的Offer

4. **DHCP ACK（确认）**：
   - 被选中的服务器发送ACK报文
   - 确认IP地址分配
   - 客户端获得IP配置

### DHCP租约

- **租期（Lease Time）**：IP地址的使用期限
- **续租**：
  - 租期过半时，客户端发送Request续租
  - 租期87.5%时，再次尝试续租
  - 租期到期后，必须重新申请
- **释放**：客户端可以主动释放IP地址

### DHCP中继（Relay）

- **问题**：DHCP广播不能跨越路由器
- **解决**：DHCP中继代理（Relay Agent）
  - 路由器配置DHCP中继
  - 将DHCP广播转发到远程DHCP服务器
  - 服务器单播回复给中继代理

### DHCP报文格式

基于BOOTP报文格式：
- 操作码（Op）：1=请求，2=响应
- 硬件类型（Htype）
- 硬件地址长度（Hlen）
- 跳数（Hops）
- 事务ID（Xid）
- 秒数（Secs）
- 标志（Flags）
- 客户端IP（Ciaddr）
- 你的IP（Yiaddr）
- 服务器IP（Siaddr）
- 网关IP（Giaddr）
- 客户端MAC（Chaddr）

## 直观理解

**DHCP就像自动分配宿舍**：
- **Discover**：新生到校广播喊"谁管宿舍分配？"
- **Offer**：宿舍管理员回复"我这有间房，301室"
- **Request**：新生说"我要301室"
- **ACK**：管理员确认"301室归你了"
- **租约**：住一年，到期续租或搬走

**记忆技巧**：
- DORA = Discover → Offer → Request → ACK
- 端口：服务器67，客户端68
- DHCP使用UDP（不是TCP）
- DHCP = "动态分配IP地址"

## 协议关联

- **DHCP与UDP**：DHCP使用UDP，端口67/68
- **DHCP与DNS**：DHCP可以分配DNS服务器地址
- **DHCP与ARP**：DHCP分配IP后，用ARP确认IP未被占用
- **DHCP与BOOTP**：DHCP基于BOOTP，兼容BOOTP
- **408考点**：
  - DHCP端口号：服务器67，客户端68
  - DHCP使用UDP
  - DORA四步过程
  - DHCP广播特性
- **陷阱**：DHCP使用UDP，不是TCP；端口67/68不可颠倒
