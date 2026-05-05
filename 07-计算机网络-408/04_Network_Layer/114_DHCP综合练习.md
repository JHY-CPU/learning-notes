# DHCP综合练习

## 核心概念
- DHCP（Dynamic Host Configuration Protocol）：动态主机配置协议
- DHCP自动为客户端分配IP地址、子网掩码、默认网关、DNS服务器等
- DHCP使用UDP传输，客户端端口68，服务器端口67

### DHCP分配的参数
- IP地址
- 子网掩码
- 默认网关
- DNS服务器地址
- 租约时间

### DHCP四步交互（DORA）
1. **Discover**（发现）：客户端广播寻找DHCP服务器
2. **Offer**（提供）：服务器响应，提供IP地址
3. **Request**（请求）：客户端请求使用该IP地址
4. **ACK**（确认）：服务器确认分配

### DHCP报文类型
| 类型 | 方向 | 说明 |
|------|------|------|
| DHCP Discover | 客户端→服务器 | 广播，寻找DHCP服务器 |
| DHCP Offer | 服务器→客户端 | 提供IP地址 |
| DHCP Request | 客户端→服务器 | 请求使用IP |
| DHCP ACK | 服务器→客户端 | 确认分配 |
| DHCP NAK | 服务器→客户端 | 拒绝请求 |
| DHCP Release | 客户端→服务器 | 释放IP地址 |

## 原理分析

### DHCP工作流程
1. 客户端启动，发送DHCP Discover（源0.0.0.0，目的255.255.255.255）
2. 服务器回复DHCP Offer（包含提供的IP地址）
3. 客户端广播DHCP Request（告知所有服务器自己的选择）
4. 选定服务器回复DHCP ACK（确认租约）
5. 租约到期前客户端需续约

### 租约续约
- 租约时间到达50%时，客户端单播发送Request续约
- 如果未收到ACK，到达87.5%时再次尝试
- 租约到期后，必须重新开始DORA流程

## 直观理解
- DHCP像"酒店入住"——前台（服务器）给你分配房间号（IP地址），有退房时间（租约）
- 四步交互像"入住流程"——问有没有空房→前台给房卡→你确认→前台确认

## 协议关联
- DHCP基于BOOTP协议发展而来
- DHCP中继代理（Relay Agent）可以让DHCP服务器不在同一子网
- DHCP用于IPv4，DHCPv6用于IPv6
- DHCP与DNS配合：DHCP分配的IP可以自动注册到DNS
