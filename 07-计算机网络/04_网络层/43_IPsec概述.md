# IPsec概述

## 核心概念
- IPsec（IP Security）是IETF制定的网络层安全协议族
- IPsec为IP数据报提供：机密性、完整性、身份认证、抗重放
- IPv6中IPsec是必选功能（IPv4中是可选的）

### IPsec协议组成
- **AH（Authentication Header，协议号51）**：
  - 提供数据完整性认证
  - 提供抗重放保护
  - 不提供加密（数据明文传输）
  - 保护IP首部（部分字段除外）
- **ESP（Encapsulating Security Payload，协议号50）**：
  - 提供数据加密（机密性）
  - 提供数据完整性认证
  - 提供抗重放保护
  - 不保护外层IP首部

### IPsec工作模式
**传输模式**：
```
原始：[IP首部][TCP/UDP][数据]
AH传输：[IP首部][AH][TCP/UDP][数据]
ESP传输：[IP首部][ESP首部][TCP/UDP][数据][ESP尾部+认证]
```

**隧道模式**：
```
AH隧道：[新IP首部][AH][原始IP首部][TCP/UDP][数据]
ESP隧道：[新IP首部][ESP首部][原始IP首部][TCP/UDP][数据][ESP尾部+认证]
```

### SA（Security Association）
- 定义了通信双方的安全参数集合
- 包含：SPI、目的IP、安全协议（AH/ESP）、算法、密钥
- 通过IKE协议自动协商建立

## 原理分析
- AH和ESP可以单独使用，也可以组合使用
- 隧道模式常用于VPN，传输模式用于端到端保护
- IKE使用UDP端口500（IKEv1）或4500（NAT穿越）

### IKE（Internet Key Exchange）
- 负责自动协商和建立SA
- IKEv1使用UDP端口500，IKEv2使用UDP端口500
- NAT穿越时使用UDP端口4500
- IKE协商过程：
  1. 协商加密算法、认证算法等
  2. 通过Diffie-Hellman交换密钥
  3. 建立IKE SA（阶段1）
  4. 建立IPsec SA（阶段2）

### AH传输模式详细格式
```
| 原始IP首部 | AH首部 | TCP | 数据 |
```
AH首部内容：下一头部(1B) | 载荷长度(1B) | 保留(2B) | SPI(4B) | 序号(4B) | 认证数据(可变)

### ESP隧道模式详细格式
```
| 新IP首部 | ESP首部 | 原始IP首部 | TCP | 数据 | ESP尾部 | ESP认证 |
```
- ESP首部：SPI(4B) | 序号(4B)
- ESP尾部：填充 | 填充长度 | 下一头部
- ESP认证：完整性校验值

### IPsec安全关联（SA）三要素
1. **SPI**（Security Parameters Index）：标识SA的索引
2. **目的IP地址**：SA对端的IP地址
3. **安全协议**：AH或ESP

### 408常考要点
- AH提供完整性+抗重放，不提供加密
- ESP提供加密+完整性+抗重放
- 隧道模式用于VPN，传输模式用于端到端
- AH不能穿越NAT（因为保护IP首部）
- IPv6中IPsec是必选的，IPv4中是可选的
- IKE使用UDP端口500

## 直观理解
- IPsec AH像"防拆封条"——保证内容没被篡改，但能看到内容
- IPsec ESP像"保险箱"——既防篡改又加密，看不到内容
- 隧道模式像"套娃"——整个原始IP数据报套在新的IP数据报中
- SA像"安全合同"——双方约定加密方式和密钥

## 协议关联
- IPsec与NAT存在兼容性问题（AH保护IP首部，穿越NAT会失败）
- ESP可以穿越NAT（不保护外层IP首部）
- VPN利用IPsec隧道模式实现远程访问或站点互联
- SSL/TLS在传输层和应用层之间提供安全，IPsec在网络层
- IPsec与IKE配合实现密钥自动协商
