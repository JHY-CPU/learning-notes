# 54_重传计时器与RTO

## 核心概念

- **重传计时器（Retransmission Timer）**：检测报文段丢失的核心机制
- **RTO（Retransmission Timeout）**：重传超时时间，基于RTT动态计算
- **RTO = SRTT + 4 × RTTVAR**
- **超时后RTO指数退避（加倍）**

### RTO计算公式

$$\text{SRTT} = (1 - \alpha) \times \text{SRTT} + \alpha \times \text{SampleRTT}, \quad \alpha = \frac{1}{8}$$

$$\text{RTTVAR} = (1 - \beta) \times \text{RTTVAR} + \beta \times |\text{SRTT} - \text{SampleRTT}|, \quad \beta = \frac{1}{4}$$

$$\text{RTO} = \text{SRTT} + 4 \times \text{RTTVAR}$$

### RTO的边界约束

- RTO下界：**1秒**（RFC推荐，实际中常用200ms或更小）
- RTO上界：**60秒**（RFC推荐）
- 首次RTO：**3秒**（RFC推荐，在收到第一个RTT样本前使用）

## 原理分析

### 重传计时器的工作流程

```
发送报文段 → 启动计时器
              ↓
        ┌─────┴─────┐
        │            │
    收到ACK       计时器超时
        │            │
   关闭计时器    重传报文段
   更新RTT估计   RTO × 2
                重启计时器
```

### 超时后的处理

1. 重传丢失的报文段
2. RTO加倍（指数退避）
3. 重启重传计时器
4. 如果持续超时，继续重传，RTO继续加倍
5. 通常重传一定次数后放弃（如Linux默认15次）

### 收到ACK后的处理

- 停止重传计时器
- 用新的RTT样本更新SRTT和RTTVAR
- 计算新的RTO
- 为下一个报文段启动新的计时器

## 直观理解

- **RTO像耐心值**：第一次没收到回复等一会儿（RTO），再没收到就多等会儿（RTO×2），越来越有耐心（指数退避），直到收到回复恢复正常

## 协议关联

- **与拥塞控制**：超时触发慢开始
- **与快速重传**：3个冗余ACK触发快速重传，不需要等RTO
- **408常见考法**：RTO计算公式（必须记住）、指数退避
