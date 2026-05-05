# 55_RTO_概念_计算

## 核心概念

- **RTO = SRTT + 4 × RTTVAR**
- **SRTT是RTT的指数加权移动平均**
- **RTTVAR是RTT偏差的指数加权移动平均**
- **超时后RTO加倍**

### 计算示例（详细步骤）

**初始**：第一个RTT样本 = 100ms

**Step 1：初始化**
$$\text{SRTT} = \text{SampleRTT} = 100\text{ms}$$
$$\text{RTTVAR} = \frac{\text{SampleRTT}}{2} = 50\text{ms}$$
$$\text{RTO} = 100 + 4 \times 50 = 300\text{ms}$$

**Step 2：第二个样本 SampleRTT = 120ms**
$$\text{SRTT} = \frac{7}{8} \times 100 + \frac{1}{8} \times 120 = 102.5\text{ms}$$
$$\text{RTTVAR} = \frac{3}{4} \times 50 + \frac{1}{4} \times |102.5 - 120| = 37.5 + 4.375 = 41.875\text{ms}$$
$$\text{RTO} = 102.5 + 4 \times 41.875 = 270\text{ms}$$

**Step 3：第三个样本 SampleRTT = 90ms**
$$\text{SRTT} = \frac{7}{8} \times 102.5 + \frac{1}{8} \times 90 = 100.9375\text{ms}$$
$$\text{RTTVAR} = \frac{3}{4} \times 41.875 + \frac{1}{4} \times |100.9375 - 90| = 31.40625 + 2.734375 = 34.14\text{ms}$$
$$\text{RTO} = 100.9375 + 4 \times 34.14 \approx 237.5\text{ms}$$

### 超时后指数退避

假设RTO=270ms时超时：
- 第1次超时：RTO = 270 × 2 = 540ms
- 第2次超时：RTO = 540 × 2 = 1080ms
- 第3次超时：RTO = 1080 × 2 = 2160ms
- 收到新ACK后：恢复为正常计算的RTO

### 408考试简化计算

如果题目给的是简化的参数，直接代入：
$$\text{RTO} = \text{SRTT} + 4 \times \text{RTTVAR}$$

## 直观理解

- **记忆公式**：RTO = 平均 + 4倍波动。波动大（网络不稳定），RTO设长些
- **指数退避**：每次超时翻倍，直到收到ACK

## 协议关联

- **与拥塞控制**：超时触发cwnd=1，ssthresh减半
- **与快速重传**：冗余ACK可以避免超时等待
- **408必背**：RTO计算公式、α=1/8、β=1/4
