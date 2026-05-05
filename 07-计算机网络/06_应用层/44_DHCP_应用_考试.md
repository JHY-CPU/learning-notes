# 45_DHCP_应用_考试

## 核心概念

- **408常考题型**：选择题为主，涉及DHCP端口号、工作过程
- **关键考点**：
  - DHCP使用UDP，服务器端口67，客户端端口68
  - DORA四步过程：Discover→Offer→Request→ACK
  - DHCP广播特性
  - DHCP中继的作用
- **易混淆点**：
  - DHCP端口：服务器67，客户端68（不可颠倒）
  - DHCP使用UDP，不使用TCP

## 原理分析

### 典型考题1：DHCP端口

**题目**：DHCP服务器使用的端口号是（  ）
A. 67
B. 68
C. 53
D. 80

**答案**：A

**解析**：
- DHCP服务器端口：67
- DHCP客户端端口：68
- DNS端口：53
- HTTP端口：80

### 典型考题2：DHCP过程

**题目**：DHCP工作过程中，客户端首先发送的是（  ）
A. DHCP Request
B. DHCP Discover
C. DHCP Offer
D. DHCP ACK

**答案**：B

**解析**：
- DORA过程：Discover→Offer→Request→ACK
- 客户端首先发送Discover广播

### 典型考题3：DHCP传输层

**题目**：DHCP使用的传输层协议是（  ）
A. TCP
B. UDP
C. ICMP
D. ARP

**答案**：B

**解析**：
- DHCP使用UDP协议
- 服务器端口67，客户端端口68

### 典型考题4：DHCP广播

**题目**：DHCP Discover报文使用的目的IP是（  ）
A. 单播IP
B. 255.255.255.255（广播）
C. DHCP服务器IP
D. 0.0.0.0

**答案**：B

**解析**：
- 客户端还没有IP地址
- 不知道DHCP服务器在哪里
- 必须使用广播（255.255.255.255）

## 直观理解

**做题技巧**：
- DHCP = UDP + 端口67(服务器)/68(客户端)
- DORA = Discover→Offer→Request→ACK
- Discover是广播（因为还不知道服务器在哪）
- DHCP中继用于跨网段

**常见错误**：
- 混淆服务器端口67和客户端端口68
- 误认为DHCP使用TCP
- 忘记DHCP Discover是广播

## 协议关联

- **DHCP与UDP**：DHCP使用UDP
- **DHCP与DNS**：DHCP分配DNS服务器地址
- **DHCP与ARP**：ARP确认IP未被占用
- **408常见组合**：DHCP + ARP + DNS = 网络配置综合题
