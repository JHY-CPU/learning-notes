# 11_重参数化技巧 (Reparameterization Trick) 推导

## 核心概念

- **问题背景**：在 VAE 中，我们需要从编码器输出的分布 $q_\phi(z|x)$ 中采样 $z$，然后通过解码器计算 $\log p_\theta(x|z)$。但采样操作不可导，无法反向传播梯度到编码器参数 $\phi$。
- **重参数化技巧**：将随机采样过程 $z \sim q_\phi(z|x)$ 重写为 $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$ 是独立于模型参数的随机噪声。这样采样操作变成了确定性变换 + 外部噪声，梯度可以流向 $\mu_\phi$ 和 $\sigma_\phi$。
- **核心思想**：从"随机变量依赖于参数"转变为"随机性来自独立的外部噪声，参数影响的是确定性的变换"。
- **适用范围**：任何需要从可参数化分布中采样并保持梯度可导的场景，不限于高斯分布——对许多分布（如 Gumbel-Softmax 用于离散分布）都有对应的重参数化方法。
- **Score Function Estimator (REINFORCE)**：如果分布不支持重参数化（如离散分布），可以使用 Score Function Estimator（也叫 REINFORCE 算法）来估计梯度，但方差通常更大。

## 数学推导

**问题形式化**：

假设我们想优化关于参数 $\phi$ 的期望：$\mathcal{L}(\phi) = \mathbb{E}_{z \sim q_\phi(z)}[f(z)]$。

直接计算梯度：$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z)}[f(z)] = \nabla_\phi \int f(z) q_\phi(z) dz$。

由于采样操作 $z \sim q_\phi(z)$ 在计算图中是离散的，梯度无法通过采样节点传播。

**重参数化的关键步骤**：

将 $z \sim q_\phi(z)$ 表示为 $z = g_\phi(\epsilon)$，其中 $\epsilon \sim p(\epsilon)$ 是一个参数无关的分布。

对于高斯分布：$q_\phi(z) = \mathcal{N}(\mu, \sigma^2)$：

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

此时：

$$
\nabla_\phi \mathbb{E}_{z \sim q_\phi(z)}[f(z)] = \nabla_\phi \mathbb{E}_{\epsilon \sim p(\epsilon)}[f(g_\phi(\epsilon))] = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]
$$

期望的梯度变成了**梯度期望**——梯度和期望可以交换顺序，从而可以采样一个 $\epsilon$ 并计算 $\nabla_\phi f(\mu + \sigma \cdot \epsilon)$。

**数学证明（标量情况）**：

原始形式：$\frac{d}{d\sigma} \int f(z) \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(z-\mu)^2}{2\sigma^2}} dz$

重参数化后：$\frac{d}{d\sigma} \int f(\mu + \sigma\epsilon) \frac{1}{\sqrt{2\pi}} e^{-\frac{\epsilon^2}{2}} d\epsilon = \int \frac{\partial f}{\partial z} \frac{\partial z}{\partial\sigma} \frac{1}{\sqrt{2\pi}} e^{-\frac{\epsilon^2}{2}} d\epsilon = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\epsilon \cdot f'(\mu + \sigma\epsilon)]$

梯度现在可以通过 $\epsilon$ 的采样来估计，且计算路径完全可微。

## 直观理解

- **重参数化技巧相当于"把随机种子作为输入"**：如果你训练一个神经网络，给它固定的随机种子 + 可学习参数，梯度当然能传播到参数上。随机种子是 $\epsilon$，参数是 $\mu$ 和 $\sigma$。
- **想象你在画靶子**：如果不重参数化，相当于每次射箭（采样）后，靶子的位置会随机移动，你无法从结果推断该往哪个方向调整瞄准。重参数化后，靶子固定了，你可以根据"箭落在靶心偏左"来调整瞄具（参数）。
- **为什么叫"重参数化"**：我们把分布 $q_\phi(z)$ 的"采样"重新参数化为"一个确定性的函数 $g_\phi$ 作用于一个固定的噪声源 $\epsilon$"，改变了问题的参数化方式。
- **没有重参数化的 VAE**：如果 VAE 没有重参数化技巧，编码器就无法通过梯度下降学习，只能使用高方差的 REINFORCE 估计器，训练将极其困难。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 重参数化采样的三种实现方式

def reparameterize_v1(mu, logvar):
    """显式写法：标准重参数化"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def reparameterize_v2(mu, logvar):
    """用 torch.normal 的写法"""
    return torch.normal(mu, torch.exp(0.5 * logvar))

def reparameterize_v3(mu, logvar):
    """最紧凑的写法"""
    return mu + torch.randn_like(mu) * logvar.mul(0.5).exp()

# 验证梯度是否可传播
def test_reparameterization_gradient():
    """验证重参数化技巧的梯度流通"""
    # 模拟编码器输出
    mu = torch.randn(10, requires_grad=True)
    logvar = torch.randn(10, requires_grad=True)
    
    # 重参数化采样
    z = reparameterize_v1(mu, logvar)
    
    # 模拟一个损失函数（如解码器的重建误差）
    loss = z.sum()
    
    # 反向传播
    loss.backward()
    
    # 验证梯度是否存在
    assert mu.grad is not None, "mu 的梯度为 None！"
    assert logvar.grad is not None, "logvar 的梯度为 None！"
    print(f"mu 的梯度: {mu.grad}")
    print(f"logvar 的梯度: {logvar.grad}")
    print("重参数化技巧成功使梯度流通！")

# 对比：没有重参数化的采样（会导致梯度断掉）
def test_without_reparameterization():
    """验证没有重参数化时梯度会断掉"""
    mu = torch.randn(10, requires_grad=True)
    
    # 错误的做法：直接从 N(mu, 1) 采样（不可导）
    z_bad = torch.normal(mu, torch.ones_like(mu))
    loss_bad = z_bad.sum()
    loss_bad.backward()
    
    # 梯度会通过 mu 传播（因为 torch.normal 的 mean 参数是支持梯度传播的）
    print(f"直接采样的 mu 梯度: {mu.grad}")

# 更复杂的例子：Gumbel-Softmax（离散重参数化）
def gumbel_softmax_sample(logits, temperature=1.0):
    """Gumbel-Softmax 重参数化：用于离散分布采样"""
    # 采样 Gumbel 噪声
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
    
    # Gumbel-Max 技巧 + softmax（使其可微）
    y = (logits + gumbel_noise) / temperature
    return F.softmax(y, dim=-1)

print("=== 验证重参数化梯度 ===")
test_reparameterization_gradient()
print("\n=== Gumbel-Softmax 示例 ===")
logits = torch.randn(4, 10, requires_grad=True)
sample = gumbel_softmax_sample(logits, temperature=0.5)
loss = sample.mean()
loss.backward()
print(f"Gumbel-Softmax 梯度存在: {logits.grad is not None}, 形状: {logits.grad.shape}")
```

## 深度学习关联

- **VAE 的核心组件**：没有重参数化技巧，VAE 就无法训练。它是 VAE 能够成功的三大创新之一（另外两个是 ELBO 和自编码架构）。
- **Gumbel-Softmax / Concrete Distribution**：将重参数化推广到离散分布，使得 VAE 可以处理离散潜变量（如文本分类），是 VQ-VAE 和文本生成模型的基础技术。
- **归一化流 (Normalizing Flows)**：重参数化技巧的更一般形式——通过一系列可逆变换将一个简单分布映射到复杂分布，每一层变换都可以视为一种重参数化。
- **随机神经网络**：在贝叶斯神经网络中，重参数化技巧被用来训练变分 Dropout 等随机正则化技术，使得网络权重的不确定性可以被学习。
