# 雅可比矩阵 (Jacobian) 与多变量链式法则

## 核心概念
- **雅可比矩阵 (Jacobian Matrix)**：对于向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，其 Jacobian 矩阵 $J \in \mathbb{R}^{m \times n}$ 定义为 $J_{ij} = \partial f_i / \partial x_j$。每一行是每个输出分量对所有输入的梯度，每一列是所有输出对某个输入的偏导。
- **Jacobian 的几何意义**：Jacobian 矩阵是向量值函数在局部的最佳线性逼近。$f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + J \Delta \mathbf{x}$。$|\det(J)|$ 衡量了映射在局部的体积缩放因子（前提是 $m = n$）。
- **多变量链式法则**：对于复合函数 $\mathbf{f} \circ \mathbf{g}$，其 Jacobian 为 $J_{f \circ g}(\mathbf{x}) = J_f(\mathbf{g}(\mathbf{x})) \cdot J_g(\mathbf{x})$，即 Jacobian 矩阵相乘。这是单变量链式法则 $(f \circ g)' = f'(g(x)) g'(x)$ 的直接推广。
- **神经网络中的 Jacobian**：每层网络相当于一个向量值函数 $\mathbf{h}^{(l)} = f^{(l)}(\mathbf{h}^{(l-1)})$，其 Jacobian 矩阵 $J^{(l)}$ 描述了输入层的微小变化如何影响输出层。
- **Jacobian 与梯度**：当 $f$ 输出是标量时（$m=1$），Jacobian 退化为梯度行向量 $\nabla f^T$。因此梯度可以视为 Jacobian 的特例。

## 数学推导
Jacobian 矩阵的定义：
$$
J = \frac{\partial(f_1, \dots, f_m)}{\partial(x_1, \dots, x_n)} = 
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

多变量链式法则：
$$
\frac{\partial \mathbf{f} \circ \mathbf{g}}{\partial \mathbf{x}}(\mathbf{x}) = 
\frac{\partial \mathbf{f}}{\partial \mathbf{g}}(\mathbf{g}(\mathbf{x})) \cdot 
\frac{\partial \mathbf{g}}{\partial \mathbf{x}}(\mathbf{x})
$$

写成求和形式：
$$
\frac{\partial f_i}{\partial x_j} = \sum_{k=1}^p \frac{\partial f_i}{\partial g_k} \cdot \frac{\partial g_k}{\partial x_j}
$$

## 直观理解
- **局部线性化的比喻**：Jacobian 就像复杂地图的"街区缩放图"。在地图上的某一点，邻域内的非线性扭曲可以用 Jacobian 近似——告诉你沿各个方向移动时，坐标变换的比例和旋转情况。
- **体积缩放**：如果 $f: \mathbb{R}^2 \to \mathbb{R}^2$ 的 Jacobian 行列式在某个区域里处处为 2，说明这个映射将该区域的面积放大了 2 倍。归一化流 (Normalizing Flows) 正是利用这一性质来精确计算概率密度的变换。
- **多变量链式法则的解读**：单变量链式法则是"导数相乘"，多变量则是"矩阵相乘"。输入的变化通过中间变量传播到输出，每个路径的贡献求和，这正是反向传播的算法原理。

## 代码示例
```python
import numpy as np

# 定义 f: R^2 -> R^2
# f1(x, y) = x^2 + y^2
# f2(x, y) = 2x + 3y
def f(xy):
    x, y = xy
    return np.array([x**2 + y**2, 2*x + 3*y])

# 解析 Jacobian
# J = [[2x, 2y], [2, 3]]
def jacobian_analytical(xy):
    x, y = xy
    return np.array([[2*x, 2*y], [2, 3]])

# 数值 Jacobian
def jacobian_numerical(f, xy, h=1e-6):
    x, y = xy
    f0 = f(xy)
    J = np.zeros((2, 2))
    # 对 x 扰动
    fx = f(np.array([x + h, y]))
    J[:, 0] = (fx - f0) / h
    # 对 y 扰动
    fy = f(np.array([x, y + h]))
    J[:, 1] = (fy - f0) / h
    return J

xy = np.array([1.0, 2.0])
J_analytical = jacobian_analytical(xy)
J_numerical = jacobian_numerical(f, xy)
print(f"解析 Jacobian:\n{J_analytical}")
print(f"数值 Jacobian:\n{J_numerical}")
print(f"误差: {np.linalg.norm(J_analytical - J_numerical):.2e}")

# 链式法则演示
def g(xy):
    x, y = xy
    return np.array([x*y, x + y])

def fg(xy):
    return f(g(xy))

# Jacobian of f∘g at (1,1): J_f(g(1,1)) * J_g(1,1)
xy0 = np.array([1.0, 1.0])
g0 = g(xy0)  # g(1,1) = (1, 2)
J_f_at_g0 = jacobian_analytical(g0)
J_g_at_xy0 = np.array([[1, 1], [1, 1]])  # g 的 Jacobian: [[y, x], [1, 1]]
J_chain = J_f_at_g0 @ J_g_at_xy0
print(f"\n链式法则 Jacobian:\n{J_chain}")

# Jacobian 行列式（体积缩放）
det_J = np.linalg.det(J_analytical)
print(f"Jacobian 行列式: {det_J:.2f}")

# 验证线性化近似
delta = np.array([0.01, 0.02])
f_exact = f(xy + delta)
f_approx = f(xy) + J_analytical @ delta
print(f"精确值: {f_exact}")
print(f"线性近似: {f_approx}")
print(f"近似误差: {np.linalg.norm(f_exact - f_approx):.2e}")
```

## 深度学习关联
- **反向传播的矩阵形式**：在深度网络中，第 $l$ 层的反向传播可以写为：
  $$\frac{\partial L}{\partial \mathbf{h}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot J^{(l)}$$
  其中 $J^{(l)}$ 是第 $l$ 层输出的 Jacobian。误差信号通过 Jacobian 矩阵从后向前传播，这正是"反向传播"名称的由来。

- **Normalizing Flows 与 Jacobian**：在归一化流中，需要计算变换 $\mathbf{z}' = f(\mathbf{z})$ 的概率密度变化：
  $$p_{\mathbf{z}'}(\mathbf{z}') = p_{\mathbf{z}}(\mathbf{z}) \cdot \left|\det\frac{\partial f^{-1}}{\partial \mathbf{z}'}\right|$$
  设计的关键是保持 Jacobian 的行列式易于计算（如三角矩阵），从而高效地训练灵活的生成模型。

- **ResNet 的 Jacobian 性质**：残差网络的映射为 $\mathbf{h}^{(l)} = \mathbf{h}^{(l-1)} + F(\mathbf{h}^{(l-1)})$，其 Jacobian 为 $I + \partial F / \partial \mathbf{h}^{(l-1)}$。单位矩阵 $I$ 提供了梯度传播的"快捷通道"，避免了梯度消失，使得训练 100 层以上的网络成为可能。

- **对抗攻击与 Jacobian**：基于 Jacobian 的对抗攻击方法（如 JSMA）计算输入对输出类别的 Jacobian 矩阵，找到对分类结果影响最大的输入维度，通过修改这些维度生成对抗样本。
