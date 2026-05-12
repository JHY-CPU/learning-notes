# DHCP综合真题

## 核心概念

### DHCP高频考点
- DHCP使用UDP传输（客户端68，服务器67）
- 四步交互：Discover → Offer → Request → ACK
- Discover和Request使用广播（目的255.255.255.255）
- DHCP提供IP地址、子网掩码、默认网关、DNS等配置

## 原理分析

### 真题一（选择题）
**Q1**: DHCP使用的传输层协议和端口是什么？
A. TCP 80 B. UDP 67/68 C. TCP 21 D. UDP 53

**答**：B。DHCP使用UDP，服务器端口67，客户端端口68。

**Q2**: DHCP的四步交互正确顺序是？
A. Discover→ACK→Request→Offer
B. Offer→Discover→ACK→Request
C. Discover→Offer→Request→ACK
D. Request→Discover→Offer→ACK

**答**：C。Discover（寻找）→Offer（提供）→Request（请求）→ACK（确认）。

**Q3**: DHCP Discover报文的目的地址是什么？
A. DHCP服务器的单播地址 B. 广播地址255.255.255.255
C. 组播地址224.0.0.1 D. 默认网关地址

**答**：B。客户端此时还不知道DHCP服务器的地址，使用广播。

### 真题二：DHCP租约
某DHCP服务器租约时间为24小时
- 12小时后（50%）：客户端单播Request续约
- 21小时后（87.5%）：再次尝试续约
- 24小时到期：如果未续约，释放IP地址

### 真题三：DHCP交互过程中的地址
| 报文 | 源IP | 目的IP |
|------|------|--------|
| Discover | 0.0.0.0 | 255.255.255.255 |
| Offer | 服务器IP | 255.255.255.255（或分配的IP） |
| Request | 0.0.0.0 | 255.255.255.255 |
| ACK | 服务器IP | 255.255.255.255（或分配的IP） |

### 真题四：DHCP中继代理
当DHCP服务器不在客户端子网时，需要DHCP中继代理（Relay Agent）：
1. 客户端发送广播Discover（目的255.255.255.255）
2. 中继代理（通常是网关路由器）收到后，转发给DHCP服务器（单播）
3. 中继代理在转发时添加"giaddr"字段（网关IP地址）
4. DHCP服务器根据giaddr确定分配哪个子网的IP

### 真题五：DHCP状态变迁
客户端状态：
1. **Init**：初始化，发送Discover
2. **Selecting**：等待Offer，选择一个服务器
3. **Requesting**：发送Request，等待ACK
4. **Bound**：获得IP，正常通信
5. **Renewing**：T1时间到，尝试续约
6. **Rebinding**：T2时间到，广播续约
7. **Expired**：租约到期，释放IP

### 真题六：概念判断
1. DHCP只能分配IP地址
2. DHCP Discover使用目的端口67
3. DHCP服务器总是用广播回复
4. DHCP租约到期后IP立即失效

**答案**：
1. **错**：DHCP还分配子网掩码、默认网关、DNS服务器等
2. **对**：DHCP Discover目的端口为67（服务器端口）
3. **错**：Offer和ACK可以用单播（如果客户端IP已知）
4. **错**：到期前客户端会尝试续约（T1=50%, T2=87.5%）

### DHCP vs BOOTP
| 特性 | BOOTP | DHCP |
|------|-------|------|
| 配置方式 | 静态绑定（手动） | 动态分配 |
| 租约 | 无 | 有（可回收） |
| 兼容性 | - | 向后兼容BOOTP |
| 配置项 | IP+引导文件 | IP+掩码+网关+DNS等 |

### 408常考要点
- DHCP使用UDP（客户端68，服务器67）
- 四步交互：Discover→Offer→Request→ACK（助记词"DO-RA"）
- Discover和Request使用广播（客户端还不知道服务器地址）
- 租约续约在50%（T1）和87.5%（T2）时间点进行
- DHCP中继代理解决跨子网DHCP问题

## 直观理解
- DHCP像"临时工位分配"——公司（服务器）给新员工分配工位（IP），员工离职（释放）后工位回收
- 四步交互像"求职流程"：投简历(Discover)→收到Offer→接受Offer(Request)→签合同(ACK)
- 租约像"租房子"——到期要续约，不续约就被收回

## 协议关联
- DHCP与ARP配合：分配IP前可能使用免费ARP检测冲突
- DHCP与DNS配合：自动注册主机名到DNS
- DHCP中继代理允许服务器不在客户端子网
- IPv6使用DHCPv6或SLAAC代替DHCP
- DHCP属于应用层协议，但服务于网络层IP配置
