# 24_SNMP_考试

## 核心概念

- **408考纲要点**：SNMP基本概念、操作类型、传输协议
- **必背知识点**：
  - SNMP使用UDP（端口161管理，162 Trap）
  - 管理者-代理模型
  - MIB是被管对象的数据库
  - Trap是代理主动向管理者发送的通知
- **SNMP操作记忆**：
  - Get：查
  - Set：改
  - Trap：报（主动告警）

## 原理分析

### 典型真题：SNMP传输协议

**题目**：SNMP使用的传输层协议和默认端口是（）
A. TCP 161 B. UDP 161 C. TCP 162 D. UDP 25

**解析**：答案B。SNMP使用UDP，管理者到代理的端口是161，代理到管理者的Trap端口是162。

### 典型真题：SNMP操作

**题目**：当被管设备发生异常时，主动向管理者发送通知的SNMP操作是（）
A. GetRequest B. SetRequest C. Trap D. GetNextRequest

**解析**：答案C。Trap是代理主动发送给管理者的告警通知，是异步的。

**题目**：管理者要查询代理的某个MIB变量值，应使用的SNMP操作是（）
A. GetRequest B. Trap C. SetRequest D. Response

**解析**：答案A。GetRequest是管理者查询代理MIB变量的操作。

### 典型真题：MIB

**题目**：MIB的含义是（）
A. 管理信息库 B. 管理信息结构 C. 简单网络管理协议 D. 网络管理系统

**解析**：答案A。MIB = Management Information Base = 管理信息库。SMI = 管理信息结构，SNMP = 协议本身。

### 典型真题：管理者-代理

**题目**：在SNMP管理模型中，代理（Agent）运行在（）
A. 网络管理中心 B. 被管设备上 C. 数据库服务器上 D. 防火墙上

**解析**：答案B。代理运行在被管设备上，管理者运行在NMS（网络管理站）上。

## 直观理解

- **答题技巧**：
  - 问"传输协议"→UDP
  - 问"端口号"→161（查询/设置），162（Trap）
  - 问"主动通知"→Trap
  - 问"查询数据"→GetRequest
  - 问"修改数据"→SetRequest
- **易错点**：
  - Trap方向：代理→管理者（不是管理者→代理）
  - SNMP用UDP不用TCP（轻量级考虑）
  - MIB不是协议，是数据库

## 协议关联

- **SNMP与UDP**：SNMP基于UDP，轻量快速
- **SNMP与MIB**：SNMP操作的对象就是MIB中的变量
- **SNMP与SMI**：SMI定义MIB中对象的格式和命名规则
- **408考法**：SNMP操作类型判断、传输协议选择、MIB概念
