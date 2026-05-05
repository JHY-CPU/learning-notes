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

## 直观理解
- IPsec AH像"防拆封条"——保证内容没被篡改，但能看到内容
- IPsec ESP像"保险箱"——既防篡改又加密，看不到内容
- 隧道模式像"套娃"——整个原始IP数据报套在新的IP数据报中

## 协议关联
- IPsec与NAT存在兼容性问题（AH保护IP首部，穿越NAT会失败）
- ESP可以穿越NAT（不保护外层IP首部）
- VPN利用IPsec隧道模式实现远程访问或站点互联
- SSL/TLS在传输层和应用层之间提供安全，IPsec在网络层
