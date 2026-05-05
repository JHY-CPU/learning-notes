# 74_无线局域网WLAN概述

## 核心概念

- **WLAN（Wireless Local Area Network）**：无线局域网，IEEE 802.11标准族
- **核心协议**：CSMA/CA（冲突避免）
- **WiFi**：WLAN的商业名称，由WiFi联盟认证

### 802.11标准族

| 标准 | 频段 | 最大速率 | 年份 |
|------|------|---------|------|
| 802.11 | 2.4 GHz | 2 Mbps | 1997 |
| 802.11b | 2.4 GHz | 11 Mbps | 1999 |
| 802.11a | 5 GHz | 54 Mbps | 1999 |
| 802.11g | 2.4 GHz | 54 Mbps | 2003 |
| 802.11n (WiFi 4) | 2.4/5 GHz | 600 Mbps | 2009 |
| 802.11ac (WiFi 5) | 5 GHz | 6.9 Gbps | 2013 |
| 802.11ax (WiFi 6) | 2.4/5 GHz | 9.6 Gbps | 2020 |

### WLAN的组网模式

| 模式 | 说明 | AP |
|------|------|-----|
| 基础设施模式 | 通过AP连接 | 有（AP = 接入点） |
| 自组织模式（Ad-hoc） | 设备直连 | 无 |

### WLAN的基本组成

- **AP（Access Point）**：接入点，相当于有线网络中的集线器/交换机
- **STA（Station）**：站点，无线终端设备
- **BSS（Basic Service Set）**：基本服务集，一个AP覆盖的范围
- **ESS（Extended Service Set）**：扩展服务集，多个BSS通过有线骨干网互联

### WLAN的安全

| 协议 | 年份 | 安全性 |
|------|------|--------|
| WEP | 1997 | 弱（已破解） |
| WPA | 2003 | 中 |
| WPA2 | 2004 | 强（AES） |
| WPA3 | 2018 | 最强 |

### 考试重点

- 802.11使用CSMA/CA，不是CSMA/CD
- AP的作用
- BSS和ESS的概念
- 隐藏终端问题

## 原理分析

### CSMA/CA在WLAN中的实现

1. 监听信道
2. 信道空闲：等待DIFS时间
3. 信道忙：执行退避算法
4. 等待DIFS后，开始发送
5. 接收方在SIFS后回复ACK

### 帧间间隔

| IFS | 全称 | 典型值 |
|-----|------|--------|
| SIFS | Short IFS | 10μs |
| PIFS | PCF IFS | 30μs |
| DIFS | DCF IFS | 50μs |

优先级：SIFS < PIFS < DIFS

### RTS/CTS的NAV

- RTS/CTS帧中携带**NAV（Network Allocation Vector）**
- NAV告知其他站点"信道将被占用多长时间"
- 其他站点根据NAV推迟自己的发送

## 直观理解

- **AP就像"无线基站"**：所有无线设备通过AP连接到有线网络
- **WLAN的不可靠性**：无线信号容易受到干扰，所以需要ACK确认机制

## 协议关联

- 802.11 MAC层使用CSMA/CA（不是CSMA/CD）
- WLAN的帧格式与以太网帧不同
- AP实现无线到有线的桥接功能
- 408考试主要考CSMA/CA和帧结构
