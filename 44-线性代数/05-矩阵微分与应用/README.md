# 05 - 矩阵微分与应用

## 1. 标量对向量的导数（梯度）

### 1.1 定义

设 $f: \mathbb{R}^n \to \mathbb{R}$，$f$ 对列向量 $\mathbf{x} = (x_1, \ldots, x_n)^T$ 的导数定义为：

$$\frac{\partial f}{\partial \mathbf{x}} = \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

这是**梯度向量**（分子布局）。注意有些文献采用分母布局（行向量形式），本文统一使用分子布局。

### 1.2 基本公式

| 函数 $f(\mathbf{x})$ | 导数 $\frac{\partial f}{\partial \mathbf{x}}$ |
|---|---|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{x} = \|\mathbf{x}\|^2$ | $2\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$（$A$ 对称） | $2A\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$（$A$ 一般） | $(A + A^T)\mathbf{x}$ |

### 1.3 方向导数

函数 $f$ 在点 $\mathbf{x}_0$ 沿方向 $\mathbf{v}$ 的方向导数为：

$$D_{\mathbf{v}} f(\mathbf{x}_0) = \nabla f(\mathbf{x}_0)^T \mathbf{v}$$

梯度方向是函数增长最快的方向。

---

## 2. 向量对向量的导数（Jacobian 矩阵）

### 2.1 定义

设 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$，$\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))^T$，则 **Jacobian 矩阵**定义为：

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}_{m \times n}$$

### 2.2 基本公式

| 函数 | Jacobian |
|---|---|
| $\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ | $A$ |
| $\mathbf{f}(\mathbf{x}) = \mathbf{x}$ | $I$ |
| $\mathbf{f}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ | $A$ |

### 2.3 链式法则

若 $\mathbf{y} = \mathbf{g}(\mathbf{u})$，$\mathbf{u} = \mathbf{h}(\mathbf{x})$，则：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$$

即 Jacobian 矩阵满足矩阵乘法的链式法则。

---

## 3. 矩阵对标量的导数

### 3.1 定义

若矩阵 $A(t)$ 的每个元素都是标量 $t$ 的函数，则：

$$\frac{dA}{dt} = \left(\frac{da_{ij}}{dt}\right)$$

### 3.2 基本公式

$$\frac{d(A(t) B(t))}{dt} = \frac{dA}{dt}B + A\frac{dB}{dt}$$

注意：一般情况下 $\frac{d(AB)}{dt} \neq A\frac{dB}{dt} + \frac{dB}{dt}A$（矩阵乘法不交换）。

$$\frac{d(A^{-1})}{dt} = -A^{-1} \frac{dA}{dt} A^{-1}$$

$$\frac{d}{dt}\det(A(t)) = \det(A(t)) \cdot \text{tr}\left(A^{-1} \frac{dA}{dt}\right)$$

---

## 4. 标量对矩阵的导数

### 4.1 定义

设 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$，$f$ 对矩阵 $X = (x_{ij})_{m \times n}$ 的导数为：

$$\frac{\partial f}{\partial X} = \begin{pmatrix} \frac{\partial f}{\partial x_{11}} & \cdots & \frac{\partial f}{\partial x_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial x_{m1}} & \cdots & \frac{\partial f}{\partial x_{mn}} \end{pmatrix}$$

导数与 $X$ 同型。

### 4.2 基本公式

| 函数 $f(X)$ | 导数 $\frac{\partial f}{\partial X}$ |
|---|---|
| $\text{tr}(AX)$ | $A^T$ |
| $\text{tr}(X^T A)$ | $A$ |
| $\text{tr}(X^T X)$ | $2X$ |
| $\text{tr}(AXB)$ | $A^T B^T$ |
| $\ln\det(X)$（$X$ 正定） | $X^{-T}$ |

---

## 5. 常用矩阵微分公式

### 5.1 Trace 相关

$$\frac{\partial}{\partial X} \text{tr}(X) = I$$

$$\frac{\partial}{\partial X} \text{tr}(X^k) = k(X^{k-1})^T$$

$$\frac{\partial}{\partial X} \text{tr}(AXBXC) = A^T X^T B^T + B^T X^T A^T \cdot C^T \quad (\text{注意矩阵顺序})$$

**通用法则**：

$$d\,\text{tr}(F(X)) = \text{tr}\left(\frac{\partial F}{\partial X}^T dX\right)$$

### 5.2 行列式相关

$$\frac{\partial}{\partial X} \det(X) = \det(X) \cdot X^{-T}$$

$$\frac{\partial}{\partial X} \ln\det(X) = X^{-T}$$

$$\frac{\partial}{\partial X} \det(AXB) = \det(AXB) \cdot A^T X^{-T} B^T$$

### 5.3 二次型

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

若 $A$ 对称：$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x}$

### 5.4 向量范数

$$\frac{\partial}{\partial \mathbf{x}} \|\mathbf{x}\| = \frac{\mathbf{x}}{\|\mathbf{x}\|}$$

$$\frac{\partial}{\partial \mathbf{x}} \|\mathbf{x}\|^2 = 2\mathbf{x}$$

$$\frac{\partial}{\partial \mathbf{x}} \|\mathbf{x}\|^p = p\|\mathbf{x}\|^{p-2} \mathbf{x}$$

---

## 6. Hessian 矩阵

### 6.1 定义

设 $f: \mathbb{R}^n \to \mathbb{R}$，**Hessian 矩阵**是 $f$ 的二阶偏导数组成的对称矩阵：

$$H = \nabla^2 f = \frac{\partial^2 f}{\partial \mathbf{x} \partial \mathbf{x}^T} = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

### 6.2 泰勒展开

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^T H \Delta\mathbf{x}$$

### 6.3 极值判定

在驻点 $\mathbf{x}^*$（$\nabla f = \mathbf{0}$）处：
- $H$ 正定 $\Rightarrow$ $\mathbf{x}^*$ 是局部极小值
- $H$ 负定 $\Rightarrow$ $\mathbf{x}^*$ 是局部极大值
- $H$ 不定 $\Rightarrow$ $\mathbf{x}^*$ 是鞍点

### 6.4 常用 Hessian 公式

| 函数 | Hessian |
|---|---|
| $\mathbf{a}^T\mathbf{x}$ | $0$ |
| $\mathbf{x}^T A\mathbf{x}$（$A$ 对称） | $2A$ |
| $-\ln f(\mathbf{x})$ | $\frac{\nabla f \nabla f^T}{f^2} - \frac{\nabla^2 f}{f}$ |

---

## 7. 在机器学习中的应用

### 7.1 线性回归的推导

**最小二乘法**：给定数据矩阵 $X \in \mathbb{R}^{m \times n}$ 和标签 $\mathbf{y} \in \mathbb{R}^m$，损失函数为：

$$L(\boldsymbol{\theta}) = \frac{1}{2} \|X\boldsymbol{\theta} - \mathbf{y}\|^2$$

求梯度：

$$\nabla_{\boldsymbol{\theta}} L = X^T(X\boldsymbol{\theta} - \mathbf{y})$$

令梯度为零，得正规方程：

$$\boldsymbol{\theta}^* = (X^T X)^{-1} X^T \mathbf{y}$$

### 7.2 梯度下降法

**更新规则**：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)$$

其中 $\eta$ 是学习率。

**矩阵形式**（线性回归）：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta X^T(\mathbf{y} - X\boldsymbol{\theta}_t)$$

**牛顿法**（使用 Hessian）：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - [H L(\boldsymbol{\theta}_t)]^{-1} \nabla L(\boldsymbol{\theta}_t)$$

对于线性回归，$H = X^T X$（常数矩阵），牛顿法一步收敛。

### 7.3 主成分分析（PCA）

给定数据中心化后的协方差矩阵 $S = \frac{1}{m} X^T X$，PCA 求解：

$$\max_{\|\mathbf{w}\|=1} \mathbf{w}^T S \mathbf{w}$$

通过拉格朗日乘数法，等价于求 $S$ 的最大特征值对应的特征向量。前 $k$ 个主成分即为 $S$ 的前 $k$ 个最大特征值对应的特征向量组成的矩阵 $W_k$。

数据降维：$\mathbf{z} = W_k^T \mathbf{x}$

---

## 8. 在深度学习中的应用

### 8.1 反向传播的矩阵形式

设神经网络第 $l$ 层的前向传播为：

$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

定义第 $l$ 层的误差：$\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$

**反向传播方程**：

$$\boldsymbol{\delta}^{(l)} = (W^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$$

**参数梯度**：

$$\frac{\partial L}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

$$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

### 8.2 批量处理的矩阵形式

对批量大小为 $B$ 的数据，前向传播可写为：

$$Z^{(l)} = A^{(l-1)} (W^{(l)})^T + \mathbf{1} (\mathbf{b}^{(l)})^T$$

其中 $A^{(l-1)} \in \mathbb{R}^{B \times n_{l-1}}$，$W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$。

这种形式充分利用了 GPU 的并行矩阵运算能力。

### 8.3 常用损失函数的梯度

**交叉熵损失**（softmax + NLL）：

$$L = -\sum_{k} y_k \log p_k, \quad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

$$\frac{\partial L}{\partial z_k} = p_k - y_k$$

这个简洁的结果正是 softmax 搭配交叉熵损失被广泛使用的原因。

**MSE 损失**：

$$L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

$$\frac{\partial L}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}} - \mathbf{y}$$

### 8.4 权重初始化与矩阵谱

网络初始化时，若权重矩阵 $W$ 的谱范数 $\|W\|_2$ 过大，会导致梯度爆炸；过小则导致梯度消失。

**Xavier 初始化**：$\text{Var}(w_{ij}) = \frac{1}{n_{in}}$（保持方差在前向和反向传播中不变）

**He 初始化**（ReLU 网络）：$\text{Var}(w_{ij}) = \frac{2}{n_{in}}$

这些初始化策略的推导本质上依赖于矩阵乘法对随机向量方差的影响：若 $W$ 的元素独立同分布，$\text{Var}(W\mathbf{x}) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(\mathbf{x})$。
