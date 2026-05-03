# 64_Temperature 参数对生成分布的影响

## 核心概念

- **Temperature (温度参数)**：在 softmax 操作前对 logits 进行缩放的参数。高温度使概率分布更"平滑"（增加随机性），低温度使分布更"尖锐"（减少随机性）。
- **Softmax 的温度公式**：$P(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$，其中 $z_i$ 是 logits，$T$ 是温度参数。
- **低温 (T < 1)**：放大概率差异。最高概率的 token 更突出，生成更确定、更有"焦点"的文本。当 $T \to 0$ 时接近贪婪解码。
- **高温 (T > 1)**：使概率分布更均匀。不同 token 的概率差距缩小，生成更多样化、更有创造性的文本。当 $T \to \infty$ 时接近均匀随机采样。
- **温度 = 1**：标准 softmax，不改变原始分布。
- **与困惑度 (Perplexity) 的关系**：低温通常降低困惑度（更确定），但可能使文本变得"安全而无聊"。高温增加困惑度（更混乱），但可能产生更有趣的输出。
- **温度调整是采样策略的组成部分**：温度通常与 Top-k/Top-p 结合使用——先调温度，再应用截断策略。
- **适当温度的选择**：任务相关——事实回答 ~0.3-0.5，创意写作 ~0.7-0.9，对话 ~0.7-1.0，诗歌/代码生成 ~0.6-0.8。

## 数学推导

**温度缩放**：给定 logits 向量 $\mathbf{z} = (z_1, \ldots, z_{|V|})$：
$$
P_T(i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}
$$

**极限行为**：
- $T \to 0^+$：$P_T(i) \to \begin{cases} 1 & \text{if } i = \arg\max_j z_j \\ 0 & \text{otherwise} \end{cases}$
- $T = 1$：标准 softmax
- $T \to \infty$：$P_T(i) \to \frac{1}{|V|}$（均匀分布）

**熵的变化**：设原始分布熵为 $H(P_1)$，温度 $T$ 下分布的熵 $H(P_T)$ 单调递增：
$$
\frac{dH(P_T)}{dT} \geq 0
$$

## 直观理解

- **Temperature 像"创造力调节旋钮"**：旋钮调到 0.3（低温），模型输出最"安全"和"可预测"的答案——就像科学家回答问题，严谨但略显保守。旋钮调到 1.0（高温），模型开始有"创意"——像诗人一样，可能会用一些新奇的表达。调到 >1.5，模型几乎是在"胡言乱语"了。
- **低温的效果**：想象一个品酒师，低温让他每次都选"最葡萄酒"的味道——稳定但无聊。低温适合需要精确性的场景（事实查询）。
- **高温的效果**：高温就像让品酒师"随缘"——可能选到意想不到的味道，偶尔惊喜，偶尔踩雷。高温适合需要创造性的场景（故事创作）。
- **温度 = 0 的风险**：$T=0$ 时模型总是选相同的最高概率词，在长文本生成中非常容易陷入重复循环（如"我我我我我"或"我喜欢我喜欢我喜欢"）。

## 代码示例

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temperature(logits, temperature):
    """带温度的 softmax"""
    return F.softmax(logits / temperature, dim=-1)

# 模拟 logits
vocab_size = 10
logits = torch.tensor([2.0, 1.5, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0])

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

print("Temperature 对概率分布的影响:")
print(f"{'温度':>6} | {'分布':>60}")
print("-" * 68)
for T in temperatures:
    probs = softmax_with_temperature(logits, T)
    probs_str = " ".join([f"{p:.3f}" for p in probs])
    print(f"{T:>6.1f} | {probs_str}")

# 计算熵
def entropy(probs):
    return -(probs * torch.log(probs + 1e-10)).sum().item()

print(f"\n{'温度':>6} {'熵':>10} {'最高概率词占比':>16}")
print("-" * 36)
for T in temperatures:
    probs = softmax_with_temperature(logits, T)
    ent = entropy(probs)
    max_prob = probs.max().item()
    print(f"{T:>6.1f} {ent:>10.4f} {max_prob:>15.2%}")

# 模拟不同温度下的采样效果
print("\n不同温度下的采样结果 (采样 10 次):")
for T in [0.2, 0.7, 1.2, 2.0]:
    samples = []
    for _ in range(10):
        probs = softmax_with_temperature(logits, T)
        sample = torch.multinomial(probs, 1).item()
        samples.append(sample)
    print(f"  T={T:.1f}: {samples}")
```

## 深度学习关联

- **推理时的关键超参数**：Temperature 是 LLM 推理时最常用的调节参数之一。在 ChatGPT 等产品中，用户可以通过调整温度控制回复的创造性——低温度给出确定答案，高温度给出创造性回答。
- **低温微调 (Distillation) 中的应用**：知识蒸馏中，教师模型的输出在高温（$T > 1$）下做 softmax 得到软标签，学生模型从这些包含类间关系信息的软标签中学习。
- **退火采样 (Annealing)**：在文本生成中，可以动态调整温度——开始时用高温探索多种可能性，然后用低温精确定位。这种"温度退火"策略在某些创作任务中效果很好。
