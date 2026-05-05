# 40_CSMA-CA_概念_考试

## 核心概念

- **408考法**：与CSMA/CD对比、隐藏/暴露终端问题、RTS/CTS机制
- **必记区别**：CSMA/CD检测冲突，CSMA/CA避免冲突
- **WiFi特性**：需要ACK、有IFS、有退避

## 原理分析

### 真题1（概念选择）

**题目**：CSMA/CA协议中，"CA"的含义是（ ）

A. Collision Accept（冲突接受）
B. Collision Avoidance（冲突避免）
C. Collision Analysis（冲突分析）
D. Channel Access（信道访问）

**答案**：B

### 真题2（对比选择）

**题目**：以下关于CSMA/CD和CSMA/CA的说法，错误的是（ ）

A. CSMA/CD用于有线网络，CSMA/CA用于无线网络
B. CSMA/CD检测冲突，CSMA/CA避免冲突
C. CSMA/CD和CSMA/CA都需要ACK确认
D. 无线网络不能使用CSMA/CD的根本原因是无法同时收发

**答案**：C

CSMA/CD不需要ACK（通过冲突检测确认），CSMA/CA需要ACK。

### 真题3（隐藏终端）

**题目**：解释隐藏终端问题及其解决方法。

**答案**：

隐藏终端：A和C都在B的通信范围内，但A和C互相不在对方范围内。A和C同时向B发送数据，导致B处冲突，但A和C都无法检测到。

解决方法：RTS/CTS机制。
- A发送RTS给B
- B回复CTS（A和C都能收到B的CTS）
- C收到CTS后推迟发送

### 真题4（帧间间隔）

**题目**：802.11中，帧间间隔的优先级顺序是（ ）

A. DIFS > PIFS > SIFS
B. SIFS > PIFS > DIFS
C. PIFS > DIFS > SIFS
D. DIFS > SIFS > PIFS

**答案**：A

DIFS最长（数据帧用），SIFS最短（ACK/CTS用）。越短优先级越高。

### 真题5（综合判断）

**题目**：以下关于无线局域网MAC协议的说法，正确的是（ ）

A. 无线局域网使用CSMA/CD协议
B. RTS/CTS机制可以完全解决冲突
C. 无线局域网必须使用ACK确认
D. 暴露终端导致信道利用率提高

**答案**：C

- A错：使用CSMA/CA
- B错：RTS/CTS减少了冲突，但不能完全消除
- C对：无线信道不可靠，必须有ACK
- D错：暴露终端降低了信道利用率

## 直观理解

- **CSMA/CA = "小心谨慎"**：不确定能不能发就不发，宁可多等
- **CSMA/CD = "大胆尝试"**：先发了再说，有冲突再处理
- **无线 vs 有线**就像"隔着墙说话" vs "面对面说话"：隔着墙没法听到对方是否也在说，只能靠约定规则

## 协议关联

- CSMA/CA与CSMA/CD是408**对比题的常客**
- WiFi的安全性（WEP/WPA/WPA2）是另一个考试热点
- CSMA/CA的效率低于CSMA/CD（因为退避和IFS的开销）
