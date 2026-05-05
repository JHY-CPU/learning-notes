# 67_DHT_考试

## 核心概念

- **408常考题型**：选择题为主，涉及DHT基本概念
- **关键考点**：
  - DHT去中心化
  - DHT用于P2P节点发现
  - DHT查找复杂度O(log N)
- **易混淆点**：
  - DHT不是协议，是数据结构
  - Kademlia是DHT的实现协议

## 原理分析

### 典型考题1：DHT特点

**题目**：关于DHT，以下说法正确的是（  ）
A. DHT有中心服务器
B. DHT查找复杂度为O(1)
C. DHT是去中心化的
D. DHT只用于BitTorrent

**答案**：C

**解析**：
- DHT无中心服务器（A错误，C正确）
- DHT查找复杂度为O(log N)（B错误）
- DHT可用于多种P2P应用（D错误）

### 典型考题2：DHT应用

**题目**：DHT常用于（  ）
A. 集中式文件共享
B. P2P节点发现
C. 邮件传输
D. Web浏览

**答案**：B

**解析**：
- DHT用于P2P网络中的节点发现
- 是Tracker的去中心化替代方案

## 直观理解

**做题技巧**：
- DHT = 去中心化
- DHT查找 = O(log N)
- DHT用于P2P

## 协议关联

- **DHT与P2P**：DHT是P2P的发现机制
- **DHT与BitTorrent**：BitTorrent使用DHT
- **408考点**：DHT的基本概念
