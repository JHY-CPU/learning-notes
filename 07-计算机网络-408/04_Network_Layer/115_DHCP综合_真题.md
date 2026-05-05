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

## 直观理解
- DHCP像"临时工位分配"——公司（服务器）给新员工分配工位（IP），员工离职（释放）后工位回收

## 协议关联
- DHCP与ARP配合：分配IP前可能使用免费ARP检测冲突
- DHCP与DNS配合：自动注册主机名到DNS
- DHCP中继代理允许服务器不在客户端子网
- IPv6使用DHCPv6或SLAAC代替DHCP
