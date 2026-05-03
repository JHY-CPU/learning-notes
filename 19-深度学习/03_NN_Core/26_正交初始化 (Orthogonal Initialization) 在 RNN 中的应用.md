# 26_正交初始化 (Orthogonal Initialization) 在 RNN 中的应用

## 核心概念

- **正交初始化的定义**：正交初始化将权重矩阵初始化为正交矩阵（即 $W^T W = I$）。对于方阵，正交矩阵的列向量两两正交且范数为 1；对于非方阵，初始化为部分正交（行或列正交）。
- **RNN 的特殊需求**：循环神经网络（RNN）的权重矩阵在时间维度上反复相乘。如果矩阵的最大奇异值大于 1，梯度会指数爆炸；如果小于 1，梯度会指数消失。正交矩阵的所有奇异值都是 1，可以最大限度缓解梯度问题。
- **奇异值谱的控制**：正交初始化确保权重矩阵的奇异值全部为 1（或接近 1），这使得信号在时间展开中既不被放大也不被衰减。这是 RNN 训练稳定性的一大进步。
- **与单位矩阵初始化的区别**：单位矩阵 $I$ 只学习对角关系，而正交矩阵允许更丰富的特征变换。正交矩阵保持了范数但不限制变换形式，保留了网络的表达能力。

## 数学推导

**正交矩阵的定义**：

方阵 $W \in \mathbb{R}^{n \times n}$ 是正交矩阵当且仅当：

$$
W^T W = W W^T = I_n
$$

**正交矩阵的性质**：

- $\det(W) = \pm 1$
- $\|Wx\|_2 = \|x\|_2$（保持欧氏范数）
- 所有奇异值为 1
- 特征值的模为 1（可以是复数）

**在 RNN 中的重要性**：

RNN 的隐藏状态更新为：

$$
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t)
$$

忽略激活函数和非线性输入，梯度反向传播 $T$ 步：

$$
\frac{\partial L}{\partial h_0} = \prod_{t=1}^{T} \left( \frac{\partial h_t}{\partial h_{t-1}} \right)^T \frac{\partial L}{\partial h_T}
$$

其中 $\frac{\partial h_t}{\partial h_{t-1}} \approx W_{hh}^T$（忽略激活函数导数）。

梯度的范数：

$$
\left\|\frac{\partial L}{\partial h_0}\right\| \leq \left\|W_{hh}\right\|_2^T \cdot \left\|\frac{\partial L}{\partial h_T}\right\|
$$

其中 $\|W_{hh}\|_2$ 是谱范数（最大奇异值）。

- 如果 $\|W_{hh}\|_2 > 1$，梯度 $\to \infty$（梯度爆炸）
- 如果 $\|W_{hh}\|_2 < 1$，梯度 $\to 0$（梯度消失）
- 如果 $\|W_{hh}\|_2 = 1$（正交矩阵），梯度范数保持不变

**生成正交矩阵的方法**：

- **QR 分解**：从随机正态分布矩阵出发，做 QR 分解，取 $Q$ 矩阵：
   $$M \sim \mathcal{N}(0, I), \quad M = QR, \quad W = Q$$

2. **Cayley 变换**：从反对称矩阵 $A$ 生成正交矩阵：
   $$W = (I - A)(I + A)^{-1}$$

- **指数映射**：从反对称矩阵 $A$ 通过矩阵指数生成：
   $$W = e^{A}$$

## 直观理解

正交初始化可以类比为"保距变换"——正交矩阵 $W$ 对向量的作用就像旋转或反射：它改变了向量的方向，但保持长度不变。这与 RNN 的需求完美契合：我们希望沿着时间步传递信息时，隐藏状态的"能量"（范数）既不会爆炸也不会消失。

在 RNN 的时间展开中，权重矩阵被反复作用于隐藏状态。如果权重矩阵不是正交的，每次相乘都会改变向量的长度——要么放大（奇异值 > 1），要么缩小（奇异值 < 1）。经过几十步后，这种缩放呈指数积累，导致数值溢出或湮灭。正交初始化从源头保证了每次"相乘"都是保距的。

想象一个"传递手电筒"的游戏：每个人收到手电筒后，把它传给下一个人。如果每传一次亮度减半（奇异值 < 1），传 10 次后人几乎看不到光了（梯度消失）；如果每传一次亮度加倍（奇异值 > 1），传几次后亮得刺眼（梯度爆炸）。正交初始化保证每次传递的亮度不变。

## 代码示例

```python
import torch
import torch.nn as nn

# 生成正交矩阵
def orthogonal_matrix(shape):
    """生成正交或半正交矩阵"""
    M = torch.randn(shape)
    if shape[0] >= shape[1]:
        Q, R = torch.linalg.qr(M)
        # 调整符号使对角线为正（数值稳定）
        d = torch.diag(R)
        Q *= d.sign().unsqueeze(0)
        return Q
    else:
        Q, R = torch.linalg.qr(M.T)
        d = torch.diag(R)
        Q *= d.sign().unsqueeze(0)
        return Q.T

# 验证正交性
W = orthogonal_matrix((64, 64))
print(f"正交矩阵 W^T W 与 I 的差异: {(W.T @ W - torch.eye(64)).norm():.6f}")
print(f"奇异值: {torch.linalg.svdvals(W)[:5]}... (全部应为 1)")

# 演示正交初始化对 RNN 梯度流动的影响
torch.manual_seed(42)

class SimpleRNN(nn.Module):
    def __init__(self, hidden_dim, init_method='orthogonal'):
        super().__init__()
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_xh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        if init_method == 'orthogonal':
            nn.init.orthogonal_(self.W_hh)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.W_hh)
        elif init_method == 'identity':
            nn.init.eye_(self.W_hh)

    def forward(self, x, steps=50):
        batch, hidden_dim = x.size(0), self.W_hh.size(0)
        h = torch.zeros(batch, hidden_dim)
        for t in range(steps):
            h = torch.tanh(x @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
        return h

hidden_dim = 128
rnn_orth = SimpleRNN(hidden_dim, 'orthogonal')
rnn_default = SimpleRNN(hidden_dim, 'xavier')

x = torch.randn(4, hidden_dim)

# 考察梯度范数
h_orth = rnn_orth(x, 50)
h_default = rnn_default(x, 50)

loss_orth = h_orth.sum()
loss_default = h_default.sum()

loss_orth.backward()
loss_default.backward()

orth_grad_norm = rnn_orth.W_hh.grad.norm().item()
default_grad_norm = rnn_default.W_hh.grad.norm().item()

print(f"\n50 步后的梯度范数:")
print(f"  正交初始化: {orth_grad_norm:.4f}")
print(f"  Xavier初始化: {default_grad_norm:.4f}")

# 隐藏状态范数随时间的变化
def track_hidden_norm(model, steps=100):
    h = torch.zeros(1, hidden_dim)
    norms = []
    for t in range(steps):
        x_t = torch.randn(1, hidden_dim) * 0.1
        h = torch.tanh(x_t @ model.W_xh.T + h @ model.W_hh.T + model.b_h)
        norms.append(h.norm().item())
    return norms

print("\n隐藏状态范数演变:")
norms_orth = track_hidden_norm(rnn_orth, 100)
norms_default = track_hidden_norm(rnn_default, 100)
print(f"  正交: 初始={norms_orth[0]:.4f}, 最终={norms_orth[-1]:.4f}")
print(f"  Xavier: 初始={norms_default[0]:.4f}, 最终={norms_default[-1]:.4f}")

# PyTorch 内置正交初始化
linear = nn.Linear(64, 64)
nn.init.orthogonal_(linear.weight)
print(f"\nPyTorch 正交初始化: W^T W ≈ I: {(linear.weight.T @ linear.weight - torch.eye(64)).norm():.6f}")
```

## 深度学习关联

- **LSTM/GRU 的标准初始化**：正交初始化是 LSTM 和 GRU 中遗忘门和输入门权重的常用初始化方法。这些门控机制需要稳定的梯度流来学习长期依赖。许多 RNN 库（如 PyTorch 的 LSTM）默认使用正交初始化。
- **深层 Transformer 中的扩展**：正交初始化的思想扩展到了深层 Transformer 中。DeepNet 等模型使用特定的初始化缩放策略（如将注意力层的输出按 $1/\sqrt{2L}$ 缩放），确保数百层 Transformer 的训练稳定性。
- **极深残差网络**：正交初始化也被用于极深的 ResNet（如 1000 层以上）。通过将每个残差块的权重初始化为接近正交，可以保证上千层的梯度流不衰减，使网络训练成为可能。
