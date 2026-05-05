# 28_TCP超时重传_概念_计算

## 核心概念

- **RTO = SRTT + 4 × RTTVAR**
- **SRTT是RTT的指数加权移动平均**
- **RTTVAR是RTT偏差的指数加权移动平均**
- **超时后RTO加倍（指数退避）**

### 计算公式汇总

$$\alpha = \frac{1}{8}, \quad \beta = \frac{1}{4}$$

$$\text{SRTT} = (1-\alpha) \cdot \text{SRTT} + \alpha \cdot \text{SampleRTT}$$

$$\text{RTTVAR} = (1-\beta) \cdot \text{RTTVAR} + \beta \cdot |\text{SRTT} - \text{SampleRTT}|$$

$$\text{RTO} = \text{SRTT} + 4 \times \text{RTTVAR}$$

## 原理分析

### 计算示例

**初始设置**：首次测量到SampleRTT = 100ms

**初始化**（第一个样本）：
- $\text{SRTT} = \text{SampleRTT} = 100\text{ms}$
- $\text{RTTVAR} = \text{SampleRTT} / 2 = 50\text{ms}$
- $\text{RTO} = \text{SRTT} + 4 \times \text{RTTVAR} = 100 + 4 \times 50 = 300\text{ms}$

**第二个样本**：SampleRTT = 120ms

$$\text{SRTT} = \frac{7}{8} \times 100 + \frac{1}{8} \times 120 = 87.5 + 15 = 102.5\text{ms}$$

$$\text{RTTVAR} = \frac{3}{4} \times 50 + \frac{1}{4} \times |100 - 120| = 37.5 + 5 = 42.5\text{ms}$$

$$\text{RTO} = 102.5 + 4 \times 42.5 = 102.5 + 170 = 272.5\text{ms}$$

**第三个样本**：SampleRTT = 90ms

$$\text{SRTT} = \frac{7}{8} \times 102.5 + \frac{1}{8} \times 90 = 89.6875 + 11.25 = 100.94\text{ms}$$

$$\text{RTTVAR} = \frac{3}{4} \times 42.5 + \frac{1}{4} \times |102.5 - 90| = 31.875 + 3.125 = 35\text{ms}$$

$$\text{RTO} = 100.94 + 4 \times 35 = 100.94 + 140 = 240.94\text{ms}$$

### 超时指数退避示例

- 第1次超时：RTO = 300ms → 下一次RTO = 600ms
- 第2次超时：RTO = 600ms → 下一次RTO = 1200ms
- 第3次超时：RTO = 1200ms → 下一次RTO = 2400ms
- 收到ACK后：恢复为正常计算的RTO

## 直观理解

- **SRTT像平均成绩**：最近的成绩权重更高，越远的成绩影响越小
- **RTTVAR像成绩波动**：波动大说明网络不稳定，RTO要设长些
- **指数退避像耐心等待**：超时一次就多等一会儿，不要反复重传加重网络负担

## 协议关联

- **与拥塞控制**：超时触发慢开始（cwnd=1, ssthresh减半）
- **与快速重传**：3个冗余ACK触发快速重传，比超时快
- **408常见考法**：给定RTT样本计算RTO（必背公式）
