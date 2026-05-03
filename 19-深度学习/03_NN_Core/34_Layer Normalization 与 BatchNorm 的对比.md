# 34_Layer Normalization 与 BatchNorm 的对比

## 核心概念

- **Layer Normalization 的定义**：Layer Normalization（LN）对单个样本的所有特征维度进行标准化，而不是像 BN 那样跨样本维度。对输入 $x \in \mathbb{R}^d$，LN 计算 $\mu = \frac{1}{d}\sum_i x_i$，$\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$，然后标准化。
- **与 BN 的核心区别**：BN 在 batch 维度（跨样本）计算统计量，LN 在特征维度（跨特征）计算统计量。BN 依赖于 batch size，LN 不依赖。BN 需要存储全局统计量用于推理，LN 不需要。
- **BN 的缺点**：BN 在小 batch 时统计量估计不准确；对 RNN 不适用（序列长度变化导致统计量不稳定）；训练和推理行为不一致。LN 解决了所有这些缺点。
- **Transformer 的首选**：LN 是 Transformer 架构的标准归一化方法。BERT、GPT、ViT 都使用 LN。这是因为 NLP 任务中序列长度变化大，且需要独立处理每个位置的 token。

## 数学推导

**LN 前向传播**：

对输入 $x \in \mathbb{R}^d$：

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

其中 $\gamma, \beta \in \mathbb{R}^d$ 是可学习参数。

**LN 反向传播**：

对输入梯度：

$$
\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\sqrt{\sigma^2 + \epsilon}} \left( \delta_i - \frac{1}{d}\sum_j \delta_j - \frac{\hat{x}_i}{d} \sum_j \delta_j \hat{x}_j \right)
$$

其中 $\delta_i = \partial L / \partial y_i$。

对可学习参数：

$$
\frac{\partial L}{\partial \gamma_i} = \sum_{batch} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
$$

$$
\frac{\partial L}{\partial \beta_i} = \sum_{batch} \frac{\partial L}{\partial y_i}
$$

**BN vs LN 梯度比较**：

BN 在 batch 维度求和：$\frac{\partial L}{\partial x} \propto \sum_{j=1}^{m} (\text{通过 batch 维度})$
LN 在特征维度求和：$\frac{\partial L}{\partial x} \propto \sum_{j=1}^{d} (\text{通过特征维度})$

**LN 的均值和方差维度**：

- BN：对 $(N, d)$ 输入，在 $N$ 维度计算均值/方差，得到 $(1, d)$
- LN：对 $(N, d)$ 输入，在 $d$ 维度计算均值/方差，得到 $(N, 1)$

**在 Transformer 中的位置**：

Post-LN（原始 Transformer）：$x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))$
Pre-LN（现代改进）：$x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))$

Pre-LN 在训练深层 Transformer 时更稳定，是目前的主流选择。

## 直观理解

BN 和 LN 的差异可以类比为"集体照 vs 个人照"：

- **BN 像是拍集体照**：调整每个人的亮度使整张照片的标准统一（跨样本标准化）。如果集体照人数太少（小 batch），统计不可靠。
- **LN 像是拍个人艺术照**：对每个人的形象独立调整（跨特征标准化）。不管多少人一起拍，每个人的处理方式不变。

在 NLP 中，一个句子中的每个 token 类似于"一个独立的个体"，各 token 的特征分布不同，LN 对每个 token 独立标准化是合理的。在 CV 中，图像的统计特性在 batch 中相对一致，BN 更有效。

LN 不需要存储推理时的全局统计量，因为它的计算只依赖当前样本自身，不依赖其他样本。这大大简化了模型部署，也是 LN 被 Transformer 广泛采用的原因之一。

## 代码示例

```python
import torch
import torch.nn as nn

# 手动实现 Layer Normalization
class LayerNormManual(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x: (batch, ..., feature_dim)
        # 在最后 normalized_shape 维上标准化
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta

# 验证手动实现与 PyTorch 内置 LN 的一致性
torch.manual_seed(42)
x = torch.randn(8, 16, 64)  # (batch, seq_len, feature)

ln_manual = LayerNormManual(64)
ln_pytorch = nn.LayerNorm(64)
ln_pytorch.weight.data = ln_manual.gamma.data.clone()
ln_pytorch.bias.data = ln_manual.beta.data.clone()

out_manual = ln_manual(x)
out_pytorch = ln_pytorch(x)
print(f"LN 输出一致: {torch.allclose(out_manual, out_pytorch, atol=1e-5)}")

# 演示 BN vs LN 的区别
x = torch.randn(8, 64)  # (batch, feature)

bn = nn.BatchNorm1d(64)
ln = nn.LayerNorm(64)

with torch.no_grad():
    bn_out = bn(x)
    ln_out = ln(x)

    print(f"\nBN 输出均值 (shape: {bn_out.shape}): {bn_out.mean():.4f}, std: {bn_out.std():.4f}")
    print(f"LN 输出均值: {ln_out.mean():.4f}, std: {ln_out.std():.4f}")

    # 每个样本的统计量
    print(f"\n每个样本的统计量:")
    for i in range(3):
        print(f"  样本 {i}: BN[{i}, :10]={x[i, :5]}")
        print(f"         LN[{i}, :10]={ln_out[i, :5]}")
        print(f"         LN mean={ln_out[i].mean():.4f}, std={ln_out[i].std():.4f}")

# 小 batch 对 BN 和 LN 的影响
print("\n小 batch 下 BN vs LN:")
for batch_size in [2, 8, 32]:
    x = torch.randn(batch_size, 64)
    bn = nn.BatchNorm1d(64)
    ln = nn.LayerNorm(64)

    bn_out = bn(x)
    ln_out = ln(x)

    # 计算输出的稳定性
    bn_std = bn_out.std()
    ln_std = ln_out.std()
    print(f"  batch={batch_size:2d}: BN std={bn_std:.4f}, LN std={ln_std:.4f}")

# 在 Transformer 风格的输入上使用 LN
class TransformerBlock(nn.Module):
    """使用 Pre-LN 的 Transformer Block"""
    def __init__(self, d_model=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048), nn.ReLU(),
            nn.Linear(2048, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN: 先 LN 再子层
        x = x + self.ffn(self.ln1(x))  # 简化版，实际还有注意力
        return x

# 使用 LN 处理变长序列
print("\nLN 处理变长序列:")
seq_lengths = [5, 10, 20]
for L in seq_lengths:
    x = torch.randn(1, L, 64)
    ln = nn.LayerNorm(64)
    out = ln(x)
    print(f"  seq_len={L:2d}: 输出 shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
```

## 深度学习关联

- **Transformer 的标准选择**：Layer Normalization 是 Transformer 系列架构的标配归一化方法。从 BERT 到 GPT-4，从 ViT 到 CLIP，所有 Transformer 模型都在注意力层和 FFN 层之前或之后使用 LN。
- **RNN 中的 LN**：LN 在 RNN/LSTM 中比 BN 更有效，因为 RNN 的序列依赖性和变长特性使 BN 难以应用。LN 对每个时间步独立标准化，解决了 RNN 训练中的梯度问题。
- **Pre-LN vs Post-LN**：原始 Transformer 使用 Post-LN（归一化在残差连接之后），但深层 Transformer 训练不稳定。Pre-LN（归一化在子层之前）在现代实现中更常见，它提供了更稳定的梯度流，使得训练数百层 Transformer 成为可能。
