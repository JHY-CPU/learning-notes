# 01_线性回归的闭式解 (Normal Equation) 推导

## 核心概念

- **线性回归**：假设目标变量 $y$ 与特征 $x_1, x_2, \dots, x_n$ 之间存在线性关系，模型形式为 $y = w_0 + w_1 x_1 + \dots + w_n x_n + \varepsilon$。
- **最小二乘法**：通过最小化预测值与真实值之间的均方误差（MSE）来求解最优参数 $w$。
- **闭式解 (Normal Equation)**：直接通过矩阵运算给出解析解 $w = (X^T X)^{-1} X^T y$，无需迭代优化。
- **设计矩阵**：将每个样本的特征向量作为行，构成矩阵 $X \in \mathbb{R}^{m \times (n+1)}$，其中第一列全为 1 对应偏置项。
- **可逆条件**：$X^T X$ 必须可逆，当特征维度大于样本数或存在多重共线性时无法直接使用。
- **对比梯度下降**：闭式解无需学习率，但对大规模数据计算 $O(n^3)$ 代价高；梯度下降适用于高维大数据场景。

## 数学推导

设训练集有 $m$ 个样本，每个样本有 $n$ 个特征。定义：
$$
X = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1n} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{m1} & \cdots & x_{mn} \end{bmatrix}, \quad y = \begin{bmatrix} y_1 \\ \vdots \\ y_m \end{bmatrix}, \quad w = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_n \end{bmatrix}
$$

均方误差损失函数为：
$$
J(w) = \frac{1}{m} \|Xw - y\|^2 = \frac{1}{m} (Xw - y)^T (Xw - y)
$$

对 $w$ 求梯度并令其为零：
$$
\nabla_w J(w) = \frac{2}{m} X^T (Xw - y) = 0
$$

整理得 $X^T X w = X^T y$，因此：
$$
\hat{w} = (X^T X)^{-1} X^T y
$$

关键推导步骤说明：展开 $(Xw - y)^T (Xw - y) = w^T X^T X w - 2 w^T X^T y + y^T y$，利用矩阵求导公式 $\nabla_w (w^T A w) = 2 A w$ 和 $\nabla_w (w^T b) = b$，即可得到梯度表达式。最终解要求 $X^T X$ 为非奇异矩阵。

## 直观理解

- **几何视角**：预测值 $\hat{y} = X w$ 是 $y$ 在 $X$ 列空间上的正交投影。Normal Equation 的本质是在寻找 $y$ 在特征空间上的最佳投影点，使得残差向量 $y - Xw$ 与 $X$ 的每一列都正交。
- **最小化距离**：想象三维空间中有一个点 $y$ 和一个平面（由 $X$ 的列张成），我们寻找平面上离 $y$ 最近的点——这就是正交投影的几何意义。
- **一次性计算 vs 迭代**：梯度下降像蒙眼下山，一步步走到谷底；Normal Equation 则像直接通过公式算出谷底坐标，一步到位。

## 代码示例

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 生成模拟数据
np.random.seed(42)
X = np.random.randn(100, 3)
true_w = np.array([2.5, -1.3, 0.8])
y = X @ true_w + np.random.randn(100) * 0.2

# 方法1：手动实现 Normal Equation
X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置列
w_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Normal Equation 解: {w_hat}")

# 方法2：使用 sklearn 验证
model = LinearRegression()
model.fit(X, y)
print(f"sklearn 系数: {model.intercept_, model.coef_}")
```

## 深度学习关联

- **线性层基础**：神经网络中的全连接层 (Linear/Dense Layer) 本质就是线性回归 $y = Wx + b$，只是后续接上了非线性激活函数。理解线性回归的闭式解有助于理解参数初始化和梯度传播的起点。
- **正则化扩展**：岭回归 (Ridge Regression) 在 Normal Equation 中加入 $L_2$ 正则项：$w = (X^T X + \lambda I)^{-1} X^T y$，这与深度学习中的权重衰减 (Weight Decay) 完全等价，体现了正则化防止过拟合的统一思想。
- **批量训练的对比**：深度学习中常用 Mini-batch SGD 而非闭式解，因为参数规模动辄百万以上，求逆操作不可行。理解 Normal Equation 的计算瓶颈 $O(n^3)$ 能帮助理解为何深度学习必须依赖迭代优化。
