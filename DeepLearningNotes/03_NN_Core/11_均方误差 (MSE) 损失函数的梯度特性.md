# 11 均方误差 (MSE) 损失函数的梯度特性

## 核心概念

- **MSE 定义**：均方误差（Mean Squared Error）定义为预测值 $\hat{y}$ 与真实值 $y$ 之差的平方的平均值：$L = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$。它是回归问题中最常用的损失函数。

- **梯度特性**：MSE 的梯度 $\partial L/\partial \hat{y} = \frac{2}{n}(\hat{y} - y)$ 与误差成正比。误差越大，梯度越大，参数更新幅度越大；误差接近零时，梯度也接近零。这提供了"误差越大，修正越猛"的直观特性。

- **对异常值的敏感性**：由于误差被平方，MSE 对异常值（outliers）极为敏感。一个离群点产生的损失可能占主导地位，导致模型为了拟合异常值而牺牲大多数正常样本的预测精度。这是 MSE 的主要缺点。

- **与最大似然估计的联系**：MSE 等价于高斯噪声假设下的负对数似然（NLL）。当假设 $y = f(x) + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$ 时，最大化似然等价于最小化 MSE。

## 数学推导

**MSE 定义**：

$$
L = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n}\|\hat{y} - y\|_2^2
$$

**梯度计算**：

对预测值 $\hat{y}_i$ 的梯度：

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

写成向量形式：

$$
\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)
$$

**对参数的梯度**（通过链式法则）：

如果 $\hat{y} = Wx + b$，则：

$$
\frac{\partial L}{\partial W} = \frac{2}{n}(\hat{y} - y) \cdot x^T
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i (\hat{y}_i - y_i)
$$

**与最大似然估计的联系**：

假设 $y_i \sim \mathcal{N}(\hat{y}_i, \sigma^2)$，即 $p(y_i|x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)$

负对数似然：

$$
-\log p(y|x) = \frac{1}{2\sigma^2} \sum_i (y_i - \hat{y}_i)^2 + \frac{n}{2}\log(2\pi\sigma^2)
$$

忽略常数项，最大化似然等价于最小化 MSE。

**凸性分析**：对于线性回归 $\hat{y} = Xw$，MSE 是 $w$ 的凸函数，具有唯一的全局最小值。但对于神经网络，由于非线性的引入，MSE 损失函数不再是凸的。

## 直观理解

MSE 就像"用弹簧连接预测点和目标点"——弹簧的弹力（梯度）正比于拉伸距离（误差）。误差越大，弹簧拉的越用力，参数调整幅度越大。但如果有一个异常点离得非常远，它的弹簧会主导整个系统，导致所有其他"正常弹簧"的调整被淹没。

从信息论的角度看，MSE 假设了高斯噪声模型。换句话说，MSE 认为误差主要来自高斯分布的随机扰动。如果实际数据中的噪声是非高斯的（如具有重尾分布），MSE 就不是最优选择。

MSE 梯度正比于误差的特性也是双刃剑：在训练初期（误差大时），梯度大有助于快速收敛；在训练后期（误差小时），梯度小有助于精细调整。但若有一个异常值保持大误差，它对应的梯度会一直主导训练。

## 代码示例

```python
import torch
import torch.nn as nn

# MSE 损失函数
mse_loss = nn.MSELoss()

# 简单示例
y_pred = torch.tensor([2.5, 0.0, 2.1, 1.4])
y_true = torch.tensor([3.0, -0.5, 2.0, 1.0])

loss = mse_loss(y_pred, y_true)
print(f"MSE Loss: {loss.item():.4f}")

# 手动计算
manual_loss = ((y_pred - y_true) ** 2).mean()
print(f"手动计算: {manual_loss.item():.4f}")

# 梯度特性演示
def mse_and_gradient(y_pred, y_true):
    diff = y_pred - y_true
    loss = (diff ** 2).mean()
    grad = 2 * diff / len(diff)  # 平均情况下的梯度
    return loss, grad

loss_val, grad_val = mse_and_gradient(y_pred, y_true)
print(f"梯度: {grad_val}")

# 异常值敏感性演示
y_pred_normal = torch.tensor([2.5, 0.5, 2.0, 1.5, 2.2])
y_true_normal = torch.tensor([3.0, 0.0, 2.0, 1.0, 2.0])

y_pred_outlier = y_pred_normal.clone()
y_true_outlier = y_true_normal.clone()
y_true_outlier[0] = 20.0  # 加入异常值

loss_normal = mse_loss(y_pred_normal, y_true_normal)
loss_outlier = mse_loss(y_pred_outlier, y_true_outlier)

print(f"\n正常数据 MSE: {loss_normal.item():.4f}")
print(f"含异常值 MSE: {loss_outlier.item():.4f}")
print(f"MSE 增幅: {(loss_outlier / loss_normal).item():.0f} 倍!")

# 线性回归中的梯度下降
torch.manual_seed(42)
x = torch.randn(100, 1)
w_true = torch.tensor([[2.0]])
b_true = torch.tensor([1.0])
y = x @ w_true + b_true + torch.randn(100, 1) * 0.5

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.01)

for epoch in range(200):
    y_pred = x @ w + b
    loss = ((y_pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"\n线性回归结果: w={w.item():.4f}, b={b.item():.4f}")
print(f"真实值: w=2.0, b=1.0")
```

## 深度学习关联

- **回归任务的标准损失**：MSE 仍然是大多数回归任务（如年龄预测、价格预测、深度估计）的默认损失函数。在计算机视觉中，深度估计网络常使用 MSE 或其变体作为损失。

- **在图像生成中的应用**：MSE 被广泛应用于图像重建任务中（如自编码器、超分辨率、去噪），但在感知质量上 MSE 最小并不一定对应视觉质量最好——这是 GAN 和感知损失（Perceptual Loss）被提出的动机之一。

- **PG-Loss 中的演进**：MSE 的平方特性在某些场景下被改进，如使用 $L_1$ 损失（MAE）或 Smooth $L_1$ 损失（Faster R-CNN 中的边界框回归）。Smooth $L_1$ 在误差大时用 $L_1$ 梯度（常数），在误差小时用 $L_2$ 梯度，兼顾了鲁棒性和收敛稳定性。
