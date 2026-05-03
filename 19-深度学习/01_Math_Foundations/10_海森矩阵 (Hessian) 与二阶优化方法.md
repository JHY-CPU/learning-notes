# 10_海森矩阵 (Hessian) 与二阶优化方法

## 核心概念

- **海森矩阵 (Hessian Matrix)**：对二次可微函数 $f: \mathbb{R}^n \to \mathbb{R}$，海森矩阵 $H$ 定义为 $H_{ij} = \partial^2 f / \partial x_i \partial x_j$。它描述了函数在某点的局部曲率信息。由于偏导次序可交换（Clairaut 定理），$H$ 是对称矩阵。
- **Hessian 的几何意义**：Hessian 的特征值和特征向量揭示了函数在局部区域的"形状"。正特征值方向向上弯曲（凸），负特征值方向向下弯曲（凹），零特征值方向是平坦的。
- **二阶泰勒展开**：
  $$f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^T H(\mathbf{x}) \Delta \mathbf{x}$$
  一阶项（梯度）描述斜率，二阶项（Hessian）描述曲率。
- **牛顿法 (Newton's Method)**：利用 Hessian 的逆调整梯度方向：
  $$\Delta \mathbf{x} = -H^{-1} \nabla f$$
  在二次函数上一步收敛，对非二次函数迭代收敛速度比梯度下降快得多（二次收敛）。
- **凸性判定**：$H \succ 0$（正定）时函数是严格凸的，任何局部极小值也是全局极小值。$H$ 不定时存在鞍点。
- **Hessian 的挑战**：对 $n$ 个参数需要计算 $n(n+1)/2$ 个二阶偏导，存储 $n \times n$ 矩阵。在深度学习中 $n$ 可达百万甚至十亿级，显式 Hessian 不可行。

## 数学推导

Hessian 矩阵的定义：
$$
H = \nabla^2 f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}
$$

牛顿法更新规则：
$$
\mathbf{x}_{t+1} = \mathbf{x}_t - H(\mathbf{x}_t)^{-1} \nabla f(\mathbf{x}_t)
$$

从泰勒展开推导牛顿法：对 $\nabla f(\mathbf{x} + \Delta \mathbf{x}) \approx \nabla f(\mathbf{x}) + H(\mathbf{x})\Delta \mathbf{x}$ 设为零，解得 $\Delta \mathbf{x} = -H^{-1}\nabla f$。

## 直观理解

- **曲率的类比**：梯度是"坡度"，Hessian 是"曲率"。如果梯度告诉你在一个斜坡上，Hessian 告诉你这个斜坡是向上弯曲（谷底）、向下弯曲（山顶）还是平坦的（鞍部）。在深度学习中，大多数局部极小值附近的 Hessian 都有正有负——这意味着它们实际上是鞍点。
- **牛顿法的直观理解**：梯度下降法就像"盲人下山"——用手杖探明当前坡度最陡的方向，沿该方向走一步。牛顿法则像"睁眼下山"——不仅看坡度，还能看到前方的曲面弯曲情况，从而直接迈向谷底。在二次曲面上，牛顿法一步到位。
- **Hessian 对角线的意义**：$\partial^2 f / \partial x_i^2$ 表示第 $i$ 个方向上的曲率。值为正且大时，该方向上函数"隆起"，优化容易收敛；值为负时，该方向上有"沟壑"，需要小心处理。

## 代码示例

```python
import numpy as np

# 定义函数 f(x, y) = x^2 + 10*y^2 (二次型，Hessian 为常数)
def f(xy):
    x, y = xy
    return x**2 + 10*y**2

def gradient(xy):
    x, y = xy
    return np.array([2*x, 20*y])

def hessian(xy):
    return np.array([[2, 0], [0, 20]])

# 比较梯度下降与牛顿法的收敛速度
import time

def gradient_descent(f, grad, x0, lr=0.1, n_iter=50):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(n_iter):
        x = x - lr * grad(x)
        path.append(x.copy())
    return x, path

def newton_method(f, grad, hess, x0, n_iter=10):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(n_iter):
        x = x - np.linalg.solve(hess(x), grad(x))
        path.append(x.copy())
    return x, path

x0 = np.array([5.0, 5.0])

# 梯度下降
gd_x, gd_path = gradient_descent(f, gradient, x0, lr=0.1, n_iter=50)
print(f"梯度下降 {len(gd_path)-1} 步后: ({gd_x[0]:.4f}, {gd_x[1]:.4f}), f={f(gd_x):.4f}")

# 牛顿法
newton_x, newton_path = newton_method(f, gradient, hessian, x0, n_iter=1)
print(f"牛顿法 1 步后: ({newton_x[0]:.4f}, {newton_x[1]:.4f}), f={f(newton_x):.4f}")

# Hessian 特征值分析
H = hessian(np.array([0, 0]))
eigvals, eigvecs = np.linalg.eigh(H)
print(f"Hessian 特征值: {eigvals} (条件数: {eigvals[-1]/eigvals[0]:.1f})")
print(f"特征向量:\n{eigvecs}")

# 随机函数的 Hessian 条件数与梯度下降收敛
np.random.seed(42)
n = 50
A = np.random.randn(n, n)
A = A.T @ A  # 正定矩阵
cond_number = np.linalg.cond(A)
print(f"\n随机正定矩阵条件数: {cond_number:.1f}")
print("条件数越大，梯度下降越慢（椭球越瘦长）")
```

## 深度学习关联

- **鞍点问题 (Saddle Point Problem)**：高维非凸优化中，鞍点而非局部极小值是主要障碍。鞍点处 $\nabla f = 0$ 但 Hessian 不定（既有正特征值又有负特征值）。在负特征值方向上，梯度为 0 但函数可以继续下降。SGD 的噪声可以帮助逃离鞍点。
- **Adam 作为近似二阶方法**：Adam 优化器使用梯度的一阶矩（动量）和二阶矩（RMS 缩放）。二阶矩实际上是对 Hessian 对角线的近似：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$，相当于用梯度平方的指数滑动平均估计 $\partial^2 L / \partial \theta_i^2$ 的对角线。
- **K-FAC (Kronecker-Factored Approximate Curvature)**：一种可扩展的近似自然梯度方法，将 Hessian 近似为克罗内克积形式 $H \approx A \otimes G$，在不显式构造 Hessian 的情况下实现二阶优化。在中小型网络上可以显著加速收敛。
- **L-BFGS**：拟牛顿法通过维护梯度差的历史记录来近似 Hessian 的逆，无需存储完整的 $H$。L-BFGS 在批处理模式下效果很好，但在 mini-batch SGD 场景下受噪声影响较大。
- **损失景观可视化**：通过将 Hessian 的最大特征值方向作为可视化轴，可以绘制损失函数在最优解附近的"剖面图"，直观地观察优化 landscape 的曲率特征。
