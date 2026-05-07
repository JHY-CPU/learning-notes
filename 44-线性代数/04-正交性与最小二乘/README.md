# 04 - 正交性与最小二乘

## 1. 内积空间的定义

### 1.1 内积的定义

在实向量空间 $V$ 上，**内积**是满足以下三条公理的二元函数 $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$：

1. **对称性**：$\langle \alpha, \beta \rangle = \langle \beta, \alpha \rangle$
2. **线性性**：$\langle k_1 \alpha_1 + k_2 \alpha_2, \beta \rangle = k_1 \langle \alpha_1, \beta \rangle + k_2 \langle \alpha_2, \beta \rangle$
3. **正定性**：$\langle \alpha, \alpha \rangle \geq 0$，且 $\langle \alpha, \alpha \rangle = 0 \Leftrightarrow \alpha = \mathbf{0}$

### 1.2 常见内积

**欧几里得内积**（$\mathbb{R}^n$ 上）：

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

**加权内积**：

$$\langle \mathbf{x}, \mathbf{y} \rangle_W = \mathbf{x}^T W \mathbf{y}$$

其中 $W$ 是正定对称矩阵。

**函数空间内积**（$C[a,b]$ 上）：

$$\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$$

### 1.3 由内积导出的概念

- **范数**：$\|\alpha\| = \sqrt{\langle \alpha, \alpha \rangle}$
- **距离**：$d(\alpha, \beta) = \|\alpha - \beta\|$
- **夹角**：$\cos\theta = \frac{\langle \alpha, \beta \rangle}{\|\alpha\| \cdot \|\beta\|}$
- **正交**：$\alpha \perp \beta \Leftrightarrow \langle \alpha, \beta \rangle = 0$

### 1.4 Cauchy-Schwarz 不等式

$$|\langle \alpha, \beta \rangle| \leq \|\alpha\| \cdot \|\beta\|$$

等号成立当且仅当 $\alpha$ 与 $\beta$ 线性相关。

### 1.5 三角不等式

$$\|\alpha + \beta\| \leq \|\alpha\| + \|\beta\|$$

---

## 2. 正交基与标准正交基

### 2.1 正交向量组

向量组 $\{\alpha_1, \ldots, \alpha_s\}$ 中任意两个不同向量正交：

$$\langle \alpha_i, \alpha_j \rangle = 0, \quad i \neq j$$

**定理**：非零正交向量组必线性无关。

**证明**：设 $k_1 \alpha_1 + \cdots + k_s \alpha_s = \mathbf{0}$，两边与 $\alpha_i$ 做内积得 $k_i \|\alpha_i\|^2 = 0$，故 $k_i = 0$。

### 2.2 正交基

若向量空间 $V$ 的一组基是正交向量组，则称为**正交基**。

### 2.3 标准正交基

若正交基的每个向量的范数都为 1，则称为**标准正交基**（规范正交基）。

标准正交基的条件：$\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij}$（Kronecker 符号）。

### 2.4 坐标的简单计算

在标准正交基 $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ 下，向量 $\alpha$ 的坐标可以直接通过内积获得：

$$\alpha = \sum_{i=1}^{n} \langle \alpha, \mathbf{e}_i \rangle \mathbf{e}_i$$

即 $x_i = \langle \alpha, \mathbf{e}_i \rangle$，无需解线性方程组。这是标准正交基的核心优势。

### 2.5 Parseval 等式

在标准正交基下，内积和范数的计算：

$$\langle \alpha, \beta \rangle = \sum_{i=1}^{n} x_i y_i = [\alpha]^T [\beta]$$

$$\|\alpha\|^2 = \sum_{i=1}^{n} x_i^2$$

---

## 3. Gram-Schmidt 正交化过程

### 3.1 正交化

设 $\{\alpha_1, \ldots, \alpha_n\}$ 是线性无关的向量组，构造正交向量组 $\{\beta_1, \ldots, \beta_n\}$：

$$\beta_1 = \alpha_1$$

$$\beta_k = \alpha_k - \sum_{j=1}^{k-1} \frac{\langle \alpha_k, \beta_j \rangle}{\langle \beta_j, \beta_j \rangle} \beta_j, \quad k = 2, 3, \ldots, n$$

### 3.2 标准化

$$\mathbf{e}_k = \frac{\beta_k}{\|\beta_k\|}$$

### 3.3 改进的 Gram-Schmidt

经典方法在数值计算中可能不稳定。**改进的 Gram-Schmidt**（Modified Gram-Schmidt）逐个正交化：

$$\beta_k^{(j)} = \beta_k^{(j-1)} - \frac{\langle \beta_k^{(j-1)}, \mathbf{e}_j \rangle}{\langle \mathbf{e}_j, \mathbf{e}_j \rangle} \mathbf{e}_j$$

每一步都使用已经正交化的结果，减少舍入误差的累积。

### 3.4 例题

将 $\alpha_1 = (1, 1, 0)^T, \alpha_2 = (1, 0, 1)^T, \alpha_3 = (0, 1, 1)^T$ 标准正交化：

- $\beta_1 = (1, 1, 0)^T$，$\mathbf{e}_1 = \frac{1}{\sqrt{2}}(1, 1, 0)^T$
- $\beta_2 = \alpha_2 - \frac{1}{2}\beta_1 = (\frac{1}{2}, -\frac{1}{2}, 1)^T$，$\mathbf{e}_2 = \frac{1}{\sqrt{6}}(1, -1, 2)^T$
- $\beta_3 = \alpha_3 - \frac{1}{2}\beta_1 - \frac{1}{3}\beta_2 = (-\frac{2}{3}, \frac{2}{3}, \frac{2}{3})^T$，$\mathbf{e}_3 = \frac{1}{\sqrt{3}}(-1, 1, 1)^T$

---

## 4. 正交矩阵的性质

### 4.1 定义

$n$ 阶实方阵 $Q$ 满足 $Q^T Q = Q Q^T = I$，即 $Q^T = Q^{-1}$，则 $Q$ 称为**正交矩阵**。

### 4.2 等价刻画

以下条件等价：
- $Q$ 是正交矩阵
- $Q$ 的列向量组是 $\mathbb{R}^n$ 的标准正交基
- $Q$ 的行向量组是 $\mathbb{R}^n$ 的标准正交基
- $\|Q\mathbf{x}\| = \|\mathbf{x}\|$（保持范数，即保距变换）
- $\langle Q\mathbf{x}, Q\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{y} \rangle$（保持内积）

### 4.3 基本性质

- $\det(Q) = \pm 1$
  - $\det(Q) = +1$：旋转（特殊正交群 $SO(n)$）
  - $\det(Q) = -1$：反射
- 正交矩阵的乘积仍是正交矩阵
- 正交矩阵的逆仍是正交矩阵
- 正交矩阵的特征值的模为 1（实特征值为 $\pm 1$）

### 4.4 二维和三维旋转矩阵

二维旋转（角度 $\theta$）：

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

三维绕 $z$ 轴旋转：

$$R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

---

## 5. 正交投影与投影矩阵

### 5.1 向量到子空间的投影

设 $W = \text{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ 是 $\mathbb{R}^n$ 的子空间，$\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ 是标准正交基。任意向量 $\mathbf{v}$ 在 $W$ 上的正交投影为：

$$\text{proj}_W(\mathbf{v}) = \sum_{i=1}^{k} \langle \mathbf{v}, \mathbf{q}_i \rangle \mathbf{q}_i = QQ^T \mathbf{v}$$

其中 $Q = [\mathbf{q}_1, \ldots, \mathbf{q}_k]$。

### 5.2 投影矩阵

$$P = QQ^T$$

**性质**：
- $P^2 = P$（幂等性）
- $P^T = P$（对称性）
- $\text{rank}(P) = \text{tr}(P) = k$
- $I - P$ 也是投影矩阵，投影到 $W^\perp$

### 5.3 投影到一般子空间

若 $W$ 的基向量组成矩阵 $A$（列向量），投影到 $W$ 的投影矩阵为：

$$P = A(A^T A)^{-1} A^T$$

前提是 $A$ 列满秩（$A^TA$ 可逆）。

### 5.4 最佳逼近性质

在 $W$ 中，$\text{proj}_W(\mathbf{v})$ 是离 $\mathbf{v}$ 最近的向量：

$$\|\mathbf{v} - \text{proj}_W(\mathbf{v})\| = \min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|$$

---

## 6. 最小二乘问题与正规方程

### 6.1 问题描述

线性方程组 $A\mathbf{x} = \mathbf{b}$（$A$ 是 $m \times n$ 矩阵，$m > n$，通常无精确解）的**最小二乘问题**是：

$$\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2$$

几何意义：在 $\text{col}(A)$ 中找离 $\mathbf{b}$ 最近的向量 $A\mathbf{x}^*$。

### 6.2 正规方程（Normal Equations）

最小二乘解满足：

$$A^T A \mathbf{x} = A^T \mathbf{b}$$

**推导**：最小二乘解要求残差 $\mathbf{r} = \mathbf{b} - A\mathbf{x}$ 与 $\text{col}(A)$ 正交，即 $A^T \mathbf{r} = \mathbf{0}$。

### 6.3 解的存在唯一性

- $A^T A$ 总是对称半正定
- 若 $A$ 列满秩（$\text{rank}(A) = n$），则 $A^T A$ 正定可逆，最小二乘解唯一：

$$\mathbf{x}^* = (A^T A)^{-1} A^T \mathbf{b}$$

- 若 $A$ 不满秩，最小二乘解不唯一，最小范数解由伪逆给出

### 6.4 线性回归中的应用

对于数据点 $(x_i, y_i)$，$i = 1, \ldots, m$，拟合直线 $y = c_0 + c_1 x$：

$$A = \begin{pmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_m \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} c_0 \\ c_1 \end{pmatrix}$$

---

## 7. QR 分解在最小二乘中的应用

### 7.1 方法

若 $A = QR$（QR 分解），则：

$$\|A\mathbf{x} - \mathbf{b}\|^2 = \|QR\mathbf{x} - \mathbf{b}\|^2 = \|R\mathbf{x} - Q^T\mathbf{b}\|^2$$

（利用正交变换保持范数的性质）

令 $\hat{\mathbf{b}} = Q^T \mathbf{b}$，问题化为求解上三角方程组：

$$R\mathbf{x} = \hat{\mathbf{b}}_1 \quad (\text{前 } n \text{ 个分量})$$

残差为 $\|\hat{\mathbf{b}}_2\|$（后 $m-n$ 个分量的范数）。

### 7.2 优势

- 数值稳定性优于正规方程（$A^TA$ 的条件数是 $A$ 的条件数的平方）
- 不需要显式计算 $A^T A$，避免精度损失
- 适合病态问题

### 7.3 与 SVD 的关系

在病态更严重的情况下，可以使用截断 SVD 进行正则化求解：

$$\mathbf{x}^* = \sum_{i=1}^{k} \frac{\mathbf{u}_i^T \mathbf{b}}{\sigma_i} \mathbf{v}_i$$

其中只保留 $\sigma_i$ 显著大于零的项，忽略小奇异值对应的分量以避免数值不稳定。

---

## 8. 正交补空间

### 8.1 定义

设 $W$ 是内积空间 $V$ 的子空间，$W$ 的**正交补**定义为：

$$W^\perp = \{\alpha \in V \mid \langle \alpha, \beta \rangle = 0, \forall \beta \in W\}$$

$W^\perp$ 也是 $V$ 的子空间。

### 8.2 核心定理

**正交分解定理**：若 $V$ 是有限维内积空间，则：

$$V = W \oplus W^\perp$$

即任意向量 $\alpha \in V$ 可唯一分解为 $\alpha = \mathbf{w} + \mathbf{w}^\perp$，其中 $\mathbf{w} \in W$，$\mathbf{w}^\perp \in W^\perp$。

### 8.3 维数公式

$$\dim(W) + \dim(W^\perp) = \dim(V)$$

### 8.4 基本子空间的正交关系

对于 $m \times n$ 矩阵 $A$：

- $\text{col}(A)^\perp = \text{null}(A^T)$（左零空间）
- $\text{null}(A)^\perp = \text{col}(A^T)$（行空间）

$$\mathbb{R}^m = \text{col}(A) \oplus \text{null}(A^T)$$

$$\mathbb{R}^n = \text{null}(A) \oplus \text{col}(A^T)$$

### 8.5 四个基本子空间

| 子空间 | 维数 | 正交补 |
|--------|------|--------|
| $\text{col}(A)$（列空间） | $\text{rank}(A)$ | $\text{null}(A^T)$ |
| $\text{null}(A)$（零空间） | $n - \text{rank}(A)$ | $\text{row}(A)$ |
| $\text{row}(A)$（行空间） | $\text{rank}(A)$ | $\text{null}(A)$ |
| $\text{null}(A^T)$（左零空间） | $m - \text{rank}(A)$ | $\text{col}(A)$ |
