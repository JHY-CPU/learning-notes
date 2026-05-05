# 25_中心极限定理在 BatchNorm 中的体现

## 核心概念

- **中心极限定理 (Central Limit Theorem, CLT)**：大量独立同分布随机变量的均值（或和）近似服从正态分布，无论原始分布形态如何。数学上：若 $X_1,\dots,X_n$ i.i.d.，则 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$。
- **CLT 的意义**：解释了为什么高斯分布在自然界和工程中无处不在。测量误差、种群身高、热力学涨落等都可以视为大量独立因素叠加的结果。
- **批量归一化 (Batch Normalization, BN)**：对每一层的激活值进行标准化：$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$，然后进行缩放和平移：$y = \gamma \hat{x} + \beta$。BN 在训练时使用当前 batch 的统计量。
- **CLT 与 BN 的联系**：BN 的有效性部分源于 CLT——大量神经元的加权和近似服从正态分布，但数据分布偏移（内部协变量偏移）会使均值和方差发生漂移。BN 通过重新标准化来"固定"分布。
- **训练与推理的差异**：训练时 BN 使用 batch 统计量（随机变量），推理时使用全局移动平均统计量（更稳定的估计）。

## 数学推导

中心极限定理（经典形式）：
设 $X_1, X_2, \dots, X_n$ 是 i.i.d. 随机变量，均值 $\mu$，方差 $\sigma^2 < \infty$。则：
$$
Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

其中 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$。

Lindberg-Levy CLT（最常用形式）：
$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

批量归一化变换：
$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

前向传播中 BN 对梯度的贡献：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{m\sqrt{\sigma_B^2 + \epsilon}} \left( m\frac{\partial L}{\partial y_i} - \sum_{j=1}^m \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_{j=1}^m \hat{x}_j \frac{\partial L}{\partial y_j} \right)
$$

## 直观理解

- **CLT 的抛硬币演示**：单次抛硬币是伯努利分布（严重非高斯），但抛 100 次硬币（二项分布）已经接近高斯。抛 1000 次几乎完美的高斯。这就是 CLT 的力量——大量小噪声叠加的结果是正态分布。
- **BN 的动机**：深度网络中，每层的输入是前一层输出的线性组合经过激活函数。由于前一层也有多个神经元，大量独立贡献的和近似正态——但训练过程中分布参数（均值和方差）会因参数更新而不断偏移。BN 用标准化操作"对齐"分布，使下一层始终接收到稳定的信号。
- **BN 为什么"有效"**：BN 确保每层输入分布具有固定的均值和方差，这样就可以使用较大的学习率而不必担心激活值爆炸或消失。同时 BN 引入了少量噪声（依赖 batch 统计量），具有轻微的正则化效果。

## 代码示例

```python
import numpy as np

# 1. 中心极限定理演示
np.random.seed(42)

# 从均匀分布（非高斯）中抽样
n_trials = 10000
sample_sizes = [1, 2, 5, 30, 100]

print("中心极限定理演示 (原始分布: Uniform[0,1]):")
for n in sample_sizes:
    # 取 n 个样本的和（重复 n_trials 次）
    sample_means = [np.mean(np.random.uniform(0, 1, n)) for _ in range(n_trials)]
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    # 理论结果：均值 = 0.5, 标准差 = sqrt(1/12)/sqrt(n)
    theoretical_std = np.sqrt(1/12) / np.sqrt(n)
    print(f"  n={n:3d}: 均值={mean_of_means:.4f} (理论=0.5), "
          f"标准差={std_of_means:.4f} (理论={theoretical_std:.4f})")

# 2. 批量归一化的简单实现
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(num_features)  # 缩放
        self.beta = np.zeros(num_features)  # 平移
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
    
    def forward(self, x):
        if self.training:
            # batch 统计量（CLT 保证了这些量的稳定性）
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # 更新全局统计量（移动平均，推理时使用）
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * batch_var
            
            # 标准化
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # 推理时使用全局统计量
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        out = self.gamma * x_normalized + self.beta
        return out

# 3. 演示 BN 的效果
np.random.seed(42)
n_samples, n_features = 1000, 32

# 生成不同分布的数据
data_uniform = np.random.uniform(-3, 3, (n_samples, n_features))
data_poisson = np.random.poisson(5, (n_samples, n_features))

bn = BatchNorm(n_features)
bn.training = True

print("\nBN 前 vs BN 后的分布统计:")
for name, data in [("均匀分布", data_uniform), ("泊松分布", data_poisson)]:
    out = bn.forward(data.astype(np.float32))
    print(f"\n{name}:")
    print(f"  输入: μ={np.mean(data):.4f}, σ={np.std(data):.4f}, 范围=[{np.min(data):.2f}, {np.max(data):.2f}]")
    print(f"  BN后: μ={np.mean(out):.4f}, σ={np.std(out):.4f}, 范围=[{np.min(out):.2f}, {np.max(out):.2f}]")

# 4. 验证 BN 的梯度传播
x = np.random.randn(32, 64)  # batch=32, features=64
bn2 = BatchNorm(64)
bn2.training = True
out = bn2.forward(x)
print(f"\nBN 输出统计 (batch_size=32):")
print(f"  每个特征都接近 N(0,1): μ≈{np.mean(out):.4f}, σ≈{np.std(out):.4f}")
```

## 深度学习关联

- **BN 的 CLT 基础与限制**：CLT 预测大量神经元的聚合近似正态，这是 BN 有效的前提。但 batch 大小影响这一近似质量——小 batch（如 batch_size=2-4）时均值方差估计不准，BN 效果下降。这也是为什么 Layer Norm 在 batch 较小时更受欢迎。
- **内部协变量偏移的缓解**：BN 最初提出的动机是缓解"内部协变量偏移"——网络参数更新导致各层输入分布的变化。后续研究表明，BN 的真正好处更多在于平滑了优化 landscape（使损失函数更接近凸函数、梯度 Lipschitz 常数更小），而非纯粹控制分布偏移。这表明 CLT 是基础但非全部。
- **BN 与其他归一化方法的对比**：根据 CLT，标准化对象不同衍生出不同方法：
  - **Batch Norm**：跨样本标准化（batch 维度），依赖 CLT 保证 batch 统计量可靠
  - **Layer Norm**：跨特征标准化（layer 维度），在 Transformer 中广泛使用
  - **Instance Norm**：单样本单通道标准化，用于图像风格迁移
  - **Group Norm**：将通道分组标准化，小 batch 下的 BN 替代方案
- **BN 的数学等价性**：有趣的是，BN 后网络的输出与 batch size 有关（因为 $\mu_B$ 和 $\sigma_B^2$ 是随机变量）。这给网络注入了隐式噪声，起到了正则化作用。在大 batch 下这种噪声较小，正则化效果减弱——这也间接利用了 CLT 的结论：batch 越大，统计量越稳定。
