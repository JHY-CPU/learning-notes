# 52_深度平衡模型 (Deep Equilibrium Models)

## 核心概念

- **DEQ 定义**：深度平衡模型（Deep Equilibrium Model, DEQ）是一种"隐式"网络。与显式堆叠 $L$ 层的前馈网络不同，DEQ 寻找一个单层的平衡点表示，相当于拥有"无限层"的深度。输出 $z^*$ 是方程 $z^* = f_\theta(z^*, x)$ 的固定点。
- **隐式层的概念**：传统网络有明确的层数，每层计算一个中间表示。DEQ 没有层数的概念——它通过求解不动点方程来得到最终的表示，等效于反复应用同一个变换直到收敛。其深度是"隐式"的。
- **使用根求解器寻找平衡点**：DEQ 不通过逐步迭代直到收敛，而是直接使用根求解器（如 Broyden 方法、Anderson 加速）找到不动点 $z^* = f_\theta(z^*, x)$。这使得"前向传播"非常高效——只需要一个收敛的固定点，而不是逐层计算。
- **通过隐函数定理反向传播**：DEQ 的反向传播不需要展开前向传播的迭代过程。通过隐函数定理（Implicit Function Theorem），可以直接从平衡点计算梯度，不需要存储任何中间状态。这实现了 $O(1)$ 的训练显存。

## 数学推导

**不动点方程**：

DEQ 的核心是一个不动点方程：

$$
z^* = f_\theta(z^*, x)
$$

其中 $z^*$ 是平衡态的隐藏状态，$f_\theta$ 是一个非线性变换（如 Transformer 的一块），$x$ 是输入。

**前向传播**：

前向传播就是求解这个不动点方程。将问题重新表述为根求解问题：

$$
g_\theta(z, x) = f_\theta(z, x) - z = 0
$$

使用根求解器（如 Broyden 方法）找到 $g_\theta(z^*, x) = 0$ 的解 $z^*$。

**反向传播（隐函数定理）**：

反向传播的目标是计算 $\partial L / \partial \theta$。关键洞察是不需要通过前向传播的迭代轨迹来反向传播（那样需要缓存所有中间状态）。

隐函数定理指出，在平衡点 $z^*$ 处：

$$
\frac{\partial z^*}{\partial \theta} = \left(I - \frac{\partial f_\theta(z^*, x)}{\partial z^*}\right)^{-1} \frac{\partial f_\theta(z^*, x)}{\partial \theta}
$$

因此：

$$
\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial z^*} \cdot \frac{\partial z^*}{\partial \theta} = \frac{\partial L}{\partial z^*} \cdot \left(I - \frac{\partial f_\theta(z^*, x)}{\partial z^*}\right)^{-1} \frac{\partial f_\theta(z^*, x)}{\partial \theta}
$$

实际实现中，不直接计算逆矩阵（$O(d^3)$ 复杂度过高），而是解一个线性系统：

$$
\left(I - \frac{\partial f_\theta}{\partial z^*}\right)^T g = \left(\frac{\partial L}{\partial z^*}\right)^T
$$

然后计算 $\frac{\partial L}{\partial \theta} = g^T \frac{\partial f_\theta}{\partial \theta}$。这个线性系统使用间接方法（如 Neumann 级数或共轭梯度法）求解。

## 直观理解

DEQ 可以类比为一个"自洽系统"：你向系统输入一个初始状态，系统反复更新自己的状态，直到状态不再变化（达到平衡）。这个最终的平衡状态就是输出。关键在于，你不需要记录每次更新的中间状态——只需要知道最终平衡状态和通往平衡状态的"方向"。

另一个类比是"无穷镜像"——两面镜子相对放置，你会看到无限重复的镜像。传统网络像是有有限面镜子（有限层），而 DEQ 像是有无限面镜子，最终收敛到一个稳定的"无限回归"图像。

DEQ 的"训练显存 $O(1)$"特性可以类比为：如果你想了解一个回声系统的特性（最终回声的模式），你不需要记录每一次回声反弹的细节（迭代过程中的中间状态），只需要知道最终稳定的回声模式（平衡点）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 简化的 DEQ 概念实现
class DEQFunction(torch.autograd.Function):
    """DEQ 的自定义自动微分函数"""
    @staticmethod
    def forward(ctx, x, *params):
        # 使用不动点迭代找到平衡点
        z = torch.zeros_like(x)  # 初始猜测
        f = DEQBlock()  # 不动点变换

        # 不动点迭代（前向）
        for _ in range(30):  # 最大迭代次数
            z_next = f(z, x)
            if torch.norm(z_next - z) < 1e-4:  # 收敛判断
                break
            z = z_next

        ctx.save_for_backward(z, x)
        ctx.f = f
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, x = ctx.saved_tensors
        f = ctx.f

        # 使用隐函数定理计算梯度
        # 简化实现：使用 Neumann 级数近似逆矩阵
        g = grad_output
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            f_out = f(z_in, x)

            # 向量-雅可比积（VJP）
            v = grad_output
            for _ in range(10):  # Neumann 级数近似
                grad_f, = torch.autograd.grad(f_out, z_in, v, retain_graph=True)
                v = grad_output + grad_f  # (I + J_f + J_f^2 + ...) * grad_output

        return v, None, None  # 简化的梯度返回

class DEQBlock(nn.Module):
    """DEQ 的不动点变换 f(z, x)"""
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z, x):
        # f(z, x) = z + net(z + x)  (residual 形式)
        return z + self.net(z + x)

class DeepEquilibriumModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.input_proj = nn.Linear(10, dim)
        self.deq_block = DEQBlock(dim)
        self.output_proj = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.input_proj(x)

        # 不动点迭代求解
        z = torch.zeros_like(h)
        for i in range(50):
            z_next = self.deq_block(z, h)
            res = torch.norm(z_next - z) / (torch.norm(z) + 1e-8)
            z = z_next
            if res < 1e-4 and i > 2:
                break

        return self.output_proj(z)

# 演示 DEQ 的训练
torch.manual_seed(42)

model = DeepEquilibriumModel(32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X = torch.randn(100, 10)
y = torch.sin(X.sum(1, keepdim=True))

print("DEQ 训练:")
for epoch in range(100):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            # 统计不动点迭代次数
            h = model.input_proj(X)
            z = torch.zeros_like(h)
            for i in range(100):
                z_next = model.deq_block(z, h)
                if torch.norm(z_next - z) < 1e-4:
                    break
                z = z_next
            print(f"  Epoch {epoch:3d}, Loss: {loss.item():.4f}, "
                  f"迭代收敛步数: {i+1}")

# 隐式梯度 vs 显式梯度的比较
print("\n隐式梯度与显式展开的梯度对比:")
z = torch.randn(4, 8, requires_grad=True)
W = torch.randn(8, 8, requires_grad=True)

def f(z):
    return torch.tanh(z @ W)

# 显式展开 5 步
z_explicit = z
for _ in range(5):
    z_explicit = f(z_explicit)
loss_explicit = z_explicit.sum()
grad_explicit = torch.autograd.grad(loss_explicit, W, retain_graph=True)[0]

# 隐式梯度（通过隐函数定理近似）
z_star = z
for _ in range(20):  # 充分迭代到收敛
    z_star = f(z_star)
loss_implicit = z_star.sum()
grad_implicit = torch.autograd.grad(loss_implicit, W, retain_graph=True)[0]

print(f"  显式 5 步梯度范数: {grad_explicit.norm():.4f}")
print(f"  隐式 20 步梯度范数: {grad_implicit.norm():.4f}")
```

## 深度学习关联

- **显存效率的革命**：DEQ 最突出的优势是训练显存 $O(1)$——无论网络等效有多深（迭代步数），训练只占用常数显存。这使得在有限显存下可以训练等效于数百层甚至数千层的模型。
- **Transformer 的 DEQ 变体**：DEQ Transformer 将 Transformer 的 $L$ 层堆叠替换为一个平衡点求解器。在保持 Transformer 性能的同时，大幅减少了参数量和显存占用。DEQ Transformer 在 GLUE 基准上取得了有竞争力的结果。
- **连续深度模型**：DEQ 与神经微分方程（Neural ODE）有密切联系。Neural ODE 将网络视为微分方程，而 DEQ 将网络视为不动点方程。两者都属于"隐式深度学习"（Implicit Deep Learning）范畴，追求突破传统"显式堆叠层"的范式。
