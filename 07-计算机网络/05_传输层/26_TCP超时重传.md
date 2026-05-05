# 27_TCP超时重传

## 核心概念

- **超时重传（Retransmission Timeout, RTO）**：发送方发送数据后启动计时器，若超时未收到确认则重传
- **RTO的计算基于RTT（Round-Trip Time，往返时间）的测量**
- **RTT**：从发送报文段到收到确认的时间
- **超时重传是TCP可靠传输的核心机制之一**

### 超时重传的基本流程

```
发送数据 → 启动重传计时器 → 等待ACK
                            ↓
                    收到ACK → 停止计时器
                    或
                    超时 → 重传数据 → 重启计时器
```

## 原理分析

### RTO的计算（RFC 6298）

**第1步：测量RTT样本**
- 每次收到ACK时，测量该报文段的RTT（称为SampleRTT）

**第2步：计算平滑RTT（SRTT）**

$$\text{SRTT} = (1 - \alpha) \times \text{SRTT} + \alpha \times \text{SampleRTT}$$

其中 $\alpha = 1/8 = 0.125$（加权移动平均）

**第3步：计算RTT偏差（RTTVAR）**

$$\text{RTTVAR} = (1 - \beta) \times \text{RTTVAR} + \beta \times |\text{SRTT} - \text{SampleRTT}|$$

其中 $\beta = 1/4 = 0.25$

**第4步：计算RTO**

$$\text{RTO} = \text{SRTT} + 4 \times \text{RTTVAR}$$

**约束条件**：
- RTO最小值 = 1秒（实际中通常为200ms）
- RTO最大值 = 60秒（RFC建议）

### 超时后的处理

- 超时后，RTO**加倍**（称为**指数退避**）
- 每次超时都将RTO设为原来的2倍，直到收到新的ACK
- 收到ACK后恢复为正常计算的RTO

### 重传二义性问题

当收到一个ACK时，可能是对原始报文段的确认，也可能是对重传报文段的确认（Karn算法）：
- **解决方法**：如果是对重传报文段的确认，不用于更新SRTT
- 超时后，RTO加倍（指数退避）

## 直观理解

- **RTO像等公交**：你记录了过去几次等公交的时间（RTT），算个平均值（SRTT），再考虑波动（RTTVAR），设个合理的等待上限（RTO）。超过了就认为公交出问题了，换其他方案（重传）

## 协议关联

- **与TCP拥塞控制的关系**：超时意味着可能发生拥塞，TCP会将ssthresh减半、cwnd重置为1（慢开始）
- **与快速重传的关系**：收到3个重复ACK时触发快速重传，不需要等待超时
- **408常见考法**：RTO计算公式（必须记住）、指数退避
