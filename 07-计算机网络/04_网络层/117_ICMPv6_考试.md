# ICMPv6考试题

## 核心概念

### 高频考点
- ICMPv6协议号（下一个首部值）为58
- ICMPv6整合了ICMPv4、ARP、IGMP三个协议的功能
- NDP（邻居发现协议）使用ICMPv6消息实现地址解析
- 路由器通告（RA）用于无状态地址自动配置（SLAAC）

### 易混淆点
- ICMPv6的"邻居请求"（NS）替代了IPv4中的ARP请求
- ICMPv6的"数据报过大"消息是路径MTU发现的核心
- ICMPv6不仅用于差错报告，还用于地址配置和邻居发现

## 原理分析

### 典型选择题
**Q1**: IPv6中，地址解析（获取邻居MAC地址）使用什么协议？
A. ARP B. RARP C. ICMPv6 NDP D. DHCPv6

**答**: C。IPv6取消了ARP，使用ICMPv6的邻居发现协议（NDP）中的NS/NA消息实现地址解析。

**Q2**: ICMPv6整合了IPv4中哪些协议的功能？
A. 仅ICMPv4 B. ICMPv4和ARP C. ICMPv4、ARP和IGMP D. 仅ARP

**答**: C。ICMPv6整合了差错报告（ICMPv4）、地址解析（ARP）和组管理（IGMP）三个功能。

**Q3**: IPv6的无状态地址自动配置（SLAAC）依赖什么消息？
A. ICMPv6回显请求 B. ICMPv6路由器通告RA C. DHCPv6 D. DNS AAAA

**答**: B。SLAAC通过ICMPv6的路由器通告（RA）消息获得网络前缀，结合EUI-64生成完整地址。

### 地址解析对比
| 步骤 | IPv4 ARP | IPv6 NDP |
|------|----------|----------|
| 请求 | ARP广播请求 | ICMPv6 NS（多播） |
| 响应 | ARP单播响应 | ICMPv6 NA（单播） |
| 目的地址 | FF:FF:FF:FF:FF:FF | 请求节点多播地址 |

### 更多考试题

**Q4**: IPv6的DAD（重复地址检测）使用什么消息？
A. ARP请求 B. ICMPv6 NS C. DHCP Discover D. ICMPv6 RS

**答**：B。DAD使用ICMPv6邻居请求（NS）来检测地址是否已被使用。

**Q5**: ICMPv6的"数据报过大"消息（类型2）的主要用途是？
A. 差错报告 B. 路径MTU发现 C. 地址解析 D. 路由器发现

**答**：B。类型2消息用于路径MTU发现，通知源端减小数据报大小。

**Q6**: MLD（组播侦听发现）在IPv6中替代了IPv4的什么协议？
A. ICMPv4 B. ARP C. IGMP D. DHCP

**答**：C。MLD替代了IPv4中的IGMP，用于管理多播组成员关系。

### ICMPv6 vs ICMPv4对比
| 功能 | ICMPv4 | ICMPv6 |
|------|--------|--------|
| 差错报告 | 支持 | 支持 |
| ping | 回显请求/应答 | 类型128/129 |
| 地址解析 | ARP（独立协议） | NDP NS/NA（ICMPv6） |
| 组管理 | IGMP（独立协议） | MLD（ICMPv6） |
| 路径MTU发现 | 类型3代码4 | 类型2 |
| 路由器发现 | ICMPv4路由器通告 | NDP RS/RA |

### NDP消息总结
| 消息类型 | 编号 | 功能 | 对应IPv4 |
|----------|------|------|----------|
| RS | 133 | 路由器请求 | ICMPv4 RS |
| RA | 134 | 路由器通告 | ICMPv4 RA |
| NS | 135 | 邻居请求 | ARP请求 |
| NA | 136 | 邻居通告 | ARP响应 |
| Redirect | 137 | 重定向 | ICMPv4重定向 |

### 408易错点
- ICMPv6不仅做差错报告，还做地址解析和组管理
- NDP的NS使用多播（请求节点多播地址），不是广播
- DAD在获得地址后、正式使用前执行
- ICMPv6的类型2（数据报过大）是路径MTU发现的关键
- MLD替代IGMP，但消息类型和工作机制不同

## 直观理解
- ICMPv6像"万能工具箱"，一个协议干了IPv4中三个协议的活
- NDP中的NS/NA像"喊话"——用多播喊一声，只有目标回应
- DAD像"入职体检"——确认地址没问题才能用
- MLD像"群主管理"——管理多播群的成员

## 协议关联
- ICMPv6与IPv6密不可分（下一个首部=58）
- SLAAC需要NDP的RA消息+EUI-64地址生成
- ping6和traceroute6都依赖ICMPv6
- 路径MTU发现是IPv6取消路由器分片的前提
