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

## 直观理解
- ICMPv6像"万能工具箱"，一个协议干了IPv4中三个协议的活
- NDP中的NS/NA像"喊话"——用多播喊一声，只有目标回应

## 协议关联
- ICMPv6与IPv6密不可分（下一个首部=58）
- SLAAC需要NDP的RA消息+EUI-64地址生成
- ping6和traceroute6都依赖ICMPv6
