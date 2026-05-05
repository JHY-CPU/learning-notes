# 30_IntServ_DiffServ_考试

## 核心概念

- **408考纲要点**：IntServ和DiffServ的基本概念和区别
- **必背对比**：
  - IntServ：每流，RSVP信令，确定性QoS，扩展性差
  - DiffServ：每类，DSCP标记，统计性QoS，扩展性好
- **DSCP相关**：IPv4 ToS字段前6位，EF=加速转发，AF=确保转发，BE=尽力而为
- **RSVP**：资源预留协议，IntServ的信令协议

## 原理分析

### 典型真题：IntServ vs DiffServ

**题目**：IntServ相对于DiffServ的主要缺点是（）
A. 不能提供QoS保障 B. 可扩展性差 C. 不需要信令协议 D. 不支持实时应用

**解析**：答案B。IntServ需要为每个流在路由器上维护状态，当流数量很大时，路由器的状态表会爆炸，可扩展性差。

### 典型真题：DiffServ工作方式

**题目**：在DiffServ模型中，复杂的功能部署在（）
A. 核心路由器 B. 边缘路由器 C. 所有路由器 D. 客户端

**解析**：答案B。DiffServ的分类、标记、整形、监管在边缘路由器完成，核心路由器只根据DSCP快速转发。

### 典型真题：DSCP

**题目**：DSCP位于IP数据包的哪个字段？（）
A. 源IP地址 B. 目的IP地址 C. ToS/服务类型字段 D. TTL字段

**解析**：答案C。DSCP使用IPv4 ToS字段（或IPv6 Traffic Class字段）的前6位。

### 典型真题：RSVP

**题目**：以下关于RSVP的描述，正确的是（）
A. RSVP是DiffServ的信令协议
B. RSVP沿数据路径预留资源
C. RSVP由发送方发起预留请求
D. RSVP不需要沿途路由器参与

**解析**：答案B。
- A错：RSVP是IntServ的信令协议
- B对：RSVP沿数据传输路径预留资源
- C错：实际中Resv消息由接收方发起（反向路径）
- D错：沿途路由器需要参与资源预留

## 直观理解

- **答题技巧**：
  - 看到"每流/逐流"→IntServ
  - 看到"每类/分类"→DiffServ
  - 看到"RSVP"→IntServ
  - 看到"DSCP"→DiffServ
  - 看到"可扩展性"→DiffServ好
  - 看到"确定性QoS"→IntServ
- **PHB记忆**：EF=快（Expedited），AF=保（Assured），BE=随（Best Effort）

## 协议关联

- **IntServ与RSVP**：RSVP在传输层（协议号46），负责预留资源
- **DiffServ与IP**：DSCP在IP头中，网络层实现
- **与流量整形**：DiffServ边缘路由器使用漏桶/令牌桶整形
- **408考法**：IntServ vs DiffServ对比、DSCP概念、RSVP作用
