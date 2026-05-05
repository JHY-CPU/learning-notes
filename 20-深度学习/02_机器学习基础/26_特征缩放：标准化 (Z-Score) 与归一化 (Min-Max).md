# 26_特征缩放：标准化 (Z-Score) 与归一化 (Min-Max)

## 核心概念

- **特征缩放**：将不同尺度的特征转换到相近的范围，是许多机器学习算法的必要预处理步骤。
- **标准化 (Z-Score Normalization)**：将特征转换为均值为 0、标准差为 1 的分布。$x' = (x - \mu) / \sigma$。不改变数据分布形状，对异常值不鲁棒。
- **归一化 (Min-Max Scaling)**：将特征线性缩放到 [0, 1] 区间。$x' = (x - x_{min}) / (x_{max} - x_{min})$。对异常值极其敏感，会将正常数据压缩到很窄的区间。
- **为什么需要缩放**：① 基于距离的模型（SVM、KNN、K-Means）假设特征尺度相同；② 梯度下降法在尺度不同时收敛缓慢（等高线呈狭长形）；③ 正则化对不同尺度的特征施加不同的惩罚强度。
- **适用范围**：标准化不假设数据分布边界，适用于可能有异常值的数据；归一化适用于数据有明确边界（如像素值 0-255）且无异常值的场景。
- **Robust Scaling**：使用中位数和四分位距 (IQR) 缩放，$x' = (x - median) / IQR$，对异常值更鲁棒。

## 数学推导

**标准化 (Z-Score)**：
$$
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$
其中 $\mu_j = \frac{1}{m} \sum_{i=1}^m x_{ij}$，$\sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^m (x_{ij} - \mu_j)^2}$。

变换后：$\mathbb{E}[x'_j] = 0$，$\text{Var}[x'_j] = 1$。

**Min-Max 归一化**：
$$
x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}
$$
变换后：$x'_{ij} \in [0, 1]$。

**两者的关系**：标准化和归一化都是线性变换（仿射变换），不会改变数据的相对距离顺序。但标准化保留了数据分布的形态，而归一化可能破坏数据的分布（尤其是存在异常值时）。

**为什么梯度下降需要特征缩放**：
考虑损失函数 $J(w) = \sum (y_i - w_1 x_{i1} - w_2 x_{i2})^2$。如果 $x_1 \in [0, 10^6]$，$x_2 \in [0, 1]$，则 $w_1$ 的梯度量级比 $w_2$ 大 $10^6$ 倍。使用统一学习率时，$w_1$ 方向的优化步伐相对于 $w_2$ 方向极不平衡，导致收敛路径呈锯齿形。

## 直观理解

- **尺子不同长度的问题**：假设你要比较两个人的身高和体重——身高用米（1.7m），体重用克（70000g）。身高的变化范围是 0.5 左右，体重是 30000 左右。不缩放时，欧氏距离完全由体重主导。特征缩放就是把"身高"和"体重"放在同一把尺子上。
- **标准化 vs 归一化的对比**：标准化像把考试成绩调整为"标准分"——看你在全班中的相对位置（高于平均几个标准差）。归一化像把考试成绩换算成百分制——看你在 0-100 中的位置。
- **为什么使用 Z-Score 时异常值会出问题**：一个离群点会显著拉偏均值和标准差，导致正常数据被压缩到接近零的范围。好比一个考了 200 分（满分 100）的学生拉高了所有人的"标准分"基线。

## 代码示例

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import load_wine

# 加载数据（不同尺度的特征）
data = load_wine()
X = data.data
print(f"原始特征范围:")
for i in range(min(4, X.shape[1])):
    print(f"  特征 {i}: [{X[:,i].min():.2f}, {X[:,i].max():.2f}]")

# 标准化
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print(f"\n标准化后 (前4特征):")
print(f"  均值: {X_std[:, :4].mean(axis=0).round(4)}")
print(f"  标准差: {X_std[:, :4].std(axis=0).round(4)}")

# Min-Max 归一化
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
print(f"\n归一化后 (前4特征):")
print(f"  范围: [{X_mm[:, :4].min():.2f}, {X_mm[:, :4].max():.2f}]")

# Robust Scaling
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
print(f"\nRobust Scaling 后 (前4特征):")
print(f"  中位数: {np.median(X_robust[:, :4], axis=0).round(4)}")
```

## 深度学习关联

- **Batch Normalization**：批量归一化是深度学习中最重要的特征缩放技术。它在每一层的激活输出上做标准化（减去批均值、除以批标准差），然后学习缩放和平移参数。这使得网络可以使用更高的学习率、减少对初始化的依赖、加速收敛。
- **Layer Normalization**：层归一化是 BatchNorm 的替代方案（在 RNN/Transformer 中更常用），对每个样本的所有特征做标准化（而非跨样本）。这在 batch size 小或序列长度变化时更稳定。
- **数据预处理的标准流程**：在深度学习中，图像数据通常做 Min-Max 归一化到 [0,1] 或标准化（减去 ImageNet 均值除以标准差）。文本和表格数据的预处理同样依赖标准化，良好的特征缩放是模型训练的前提。
