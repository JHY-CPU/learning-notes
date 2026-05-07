# 03 - 特征值与矩阵分解

## 1. 特征值与特征向量的定义

### 1.1 定义

设 $A$ 是 $n$ 阶方阵，若存在非零向量 $\mathbf{x}$ 和标量 $\lambda$ 使得：

$$A\mathbf{x} = \lambda \mathbf{x}$$

则 $\lambda$ 称为 $A$ 的**特征值**，$\mathbf{x}$ 称为属于 $\lambda$ 的**特征向量**。

### 1.2 几何意义

特征向量是在线性变换 $A$ 作用下方向不变的向量（仅发生伸缩），伸缩因子即为特征值。

### 1.3 特征方程

$A\mathbf{x} = \lambda \mathbf{x}$ 等价于 $(A - \lambda I)\mathbf{x} = \mathbf{0}$ 有非零解，即：

$$\det(A - \lambda I) = 0$$

这就是**特征方程**，展开后得到 $\lambda$ 的 $n$ 次多项式。

---

## 2. 特征多项式与求解方法

### 2.1 特征多项式

$$p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + (-1)^{n-1}(\text{tr}A)\lambda^{n-1} + \cdots + \det(A)$$

特征多项式的根即为特征值。

### 2.2 求解步骤

1. 计算特征多项式 $p(\lambda) = \det(A - \lambda I)$
2. 求解特征方程 $p(\lambda) = 0$，得特征值 $\lambda_1, \ldots, \lambda_k$
3. 对每个 $\lambda_i$，求解 $(A - \lambda_i I)\mathbf{x} = \mathbf{0}$ 的基础解系
4. 基础解系的线性组合即为 $\lambda_i$ 对应的全部特征向量

### 2.3 代数重数与几何重数

- **代数重数**：$\lambda_i$ 作为特征多项式根的重数
- **几何重数**：$\dim(E_{\lambda_i})$，其中 $E_{\lambda_i} = \ker(A - \lambda_i I)$ 是特征子空间

**关系**：$1 \leq \text{几何重数} \leq \text{代数重数}$

---

## 3. 特征值的性质

### 3.1 迹与行列式

$$\sum_{i=1}^{n} \lambda_i = \text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

$$\prod_{i=1}^{n} \lambda_i = \det(A)$$

### 3.2 矩阵运算的特征值

| 运算 | 特征值 |
|------|--------|
| $A^k$ | $\lambda^k$ |
| $A^{-1}$ | $1/\lambda$（$\lambda \neq 0$） |
| $A^T$ | $\lambda$（与 $A$ 相同） |
| $A + cI$ | $\lambda + c$ |
| $\text{adj}(A)$ | $\det(A)/\lambda$（$\lambda \neq 0$） |

### 3.3 相似矩阵的特征值

若 $A \sim B$（即 $B = P^{-1}AP$），则 $A$ 和 $B$ 有相同的特征多项式，从而有相同的特征值。

**注意**：特征值相同不一定相似。

### 3.4 不同特征值对应的特征向量线性无关

若 $\lambda_1, \ldots, \lambda_k$ 是 $A$ 的互异特征值，对应的特征向量 $\mathbf{x}_1, \ldots, \mathbf{x}_k$ 线性无关。

---

## 4. 矩阵对角化的条件与方法

### 4.1 对角化

$n$ 阶方阵 $A$ 可对角化，若存在可逆矩阵 $P$ 使得 $P^{-1}AP = D$（对角矩阵），其中 $D$ 的对角元素为 $A$ 的特征值。

### 4.2 充要条件

$A$ 可对角化 $\Leftrightarrow$ $A$ 有 $n$ 个线性无关的特征向量

等价条件：
- 每个特征值的**几何重数等于代数重数**
- $A$ 有 $n$ 个互异特征值（充分但不必要条件）

### 4.3 对角化步骤

1. 求出所有特征值 $\lambda_1, \ldots, \lambda_k$
2. 对每个 $\lambda_i$，求特征子空间 $E_{\lambda_i}$ 的基
3. 验证：所有基的向量个数之和是否为 $n$
4. 若是，以这些特征向量为列构成 $P$，使 $P^{-1}AP = D$

### 4.4 应用

若 $A = PDP^{-1}$，则 $A^k = PD^kP^{-1}$，可用于高效计算矩阵幂。

---

## 5. 实对称矩阵的对角化

### 5.1 基本定理

实对称矩阵 $A = A^T$ 的特征值全是实数。

### 5.2 正交对角化

实对称矩阵必可正交对角化：存在正交矩阵 $Q$ 使得：

$$Q^T A Q = \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$$

### 5.3 谱定理

不同特征值对应的特征向量相互正交。

同一特征值的特征子空间可以通过 Gram-Schmidt 正交化得到标准正交基。

### 5.4 正交对角化步骤

1. 求出所有特征值
2. 对每个特征值求特征子空间的基
3. 对每个特征子空间的基进行 Gram-Schmidt 正交化
4. 所有标准正交特征向量组成正交矩阵 $Q$

---

## 6. Jordan 标准形

### 6.1 Jordan 块

$$J_k(\lambda) = \begin{pmatrix} \lambda & 1 & & \\ & \lambda & \ddots & \\ & & \ddots & 1 \\ & & & \lambda \end{pmatrix}_{k \times k}$$

### 6.2 Jordan 标准形定理

任何复方阵 $A$ 都相似于一个 Jordan 矩阵：

$$A = P J P^{-1}, \quad J = \text{diag}(J_{k_1}(\lambda_1), J_{k_2}(\lambda_2), \ldots)$$

Jordan 标准形在同构意义下唯一（不计 Jordan 块的排列顺序）。

### 6.3 求 Jordan 标准形

对每个特征值 $\lambda$：
- $J(\lambda)$ 的个数 = $\dim(\ker(A - \lambda I))$（几何重数）
- 所有 Jordan 块的阶数之和 = $\lambda$ 的代数重数

### 6.4 广义特征向量

满足 $(A - \lambda I)^k \mathbf{x} = \mathbf{0}$（对某个 $k$）但 $(A - \lambda I)^{k-1} \mathbf{x} \neq \mathbf{0}$ 的向量称为广义特征向量。

$A$ 可对角化 $\Leftrightarrow$ 每个特征值的几何重数等于代数重数 $\Leftrightarrow$ 没有广义特征向量。

---

## 7. LU 分解

### 7.1 Doolittle 分解

将矩阵 $A$ 分解为单位下三角矩阵 $L$ 和上三角矩阵 $U$ 的乘积：

$$A = LU$$

其中 $L$ 的对角线元素为 1。

**计算公式**（对 $n$ 阶矩阵）：

- $U$ 的第 $j$ 行：$u_{rj} = a_{rj} - \sum_{k=1}^{r-1} l_{rk} u_{kj}$
- $L$ 的第 $i$ 列：$l_{ir} = \frac{1}{u_{rr}}\left(a_{ir} - \sum_{k=1}^{r-1} l_{ik} u_{kr}\right)$

### 7.2 Crout 分解

$U$ 的对角线元素为 1，$L$ 为一般下三角矩阵。

### 7.3 带行交换的 PLU 分解

$$PA = LU$$

其中 $P$ 是置换矩阵。这是数值计算中更稳定的版本。

---

## 8. QR 分解

### 8.1 定义

任何实矩阵 $A_{m \times n}$（$m \geq n$，列满秩）可分解为：

$$A = QR$$

其中 $Q$ 是 $m \times n$ 列正交矩阵（$Q^TQ = I_n$），$R$ 是 $n \times n$ 上三角矩阵。

### 8.2 Gram-Schmidt 正交化法

设 $A$ 的列为 $\mathbf{a}_1, \ldots, \mathbf{a}_n$：

1. $\mathbf{u}_1 = \mathbf{a}_1$
2. $\mathbf{u}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} \frac{\langle \mathbf{a}_k, \mathbf{u}_j \rangle}{\langle \mathbf{u}_j, \mathbf{u}_j \rangle} \mathbf{u}_j$

然后单位化：$\mathbf{q}_k = \mathbf{u}_k / \|\mathbf{u}_k\|$。

$Q = [\mathbf{q}_1, \ldots, \mathbf{q}_n]$，$R$ 的元素为 $r_{jk} = \langle \mathbf{q}_j, \mathbf{a}_k \rangle$（$j \leq k$）。

### 8.3 Householder 变换法

利用 Householder 反射将 $A$ 逐步化为上三角矩阵，数值稳定性优于 Gram-Schmidt。

### 8.4 应用

- 求解最小二乘问题
- 计算特征值（QR 迭代算法）
- 正交化过程

---

## 9. 奇异值分解（SVD）

### 9.1 定义

任何 $m \times n$ 实矩阵 $A$ 可分解为：

$$A = U \Sigma V^T$$

其中：
- $U$ 是 $m \times m$ 正交矩阵（左奇异向量）
- $\Sigma$ 是 $m \times n$ 对角矩阵，对角元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$（奇异值）
- $V$ 是 $n \times n$ 正交矩阵（右奇异向量）
- $r = \text{rank}(A)$

### 9.2 求法

- $\sigma_i = \sqrt{\lambda_i(A^TA)}$，其中 $\lambda_i$ 是 $A^TA$ 的非零特征值
- $V$ 的列是 $A^TA$ 的特征向量
- $U$ 的列可由 $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$ 得到

### 9.3 性质

- $\|A\|_2 = \sigma_1$（最大奇异值等于谱范数）
- $\|A\|_F = \sqrt{\sum \sigma_i^2}$
- $\text{rank}(A) = $ 非零奇异值的个数
- $\text{cond}(A) = \sigma_1 / \sigma_r$（条件数）

### 9.4 应用

**降维（截断 SVD）**：

用前 $k$ 个奇异值近似 $A$：$A \approx A_k = U_k \Sigma_k V_k^T$

- 最优性：$A_k$ 是在 Frobenius 范数下对 $A$ 的最佳秩 $k$ 近似（Eckart-Young 定理）
- **PCA** 中，SVD 等价于对数据协方差矩阵的特征值分解

**推荐系统**：

用户-物品评分矩阵 $R$ 通常是稀疏的。SVD 将 $R$ 分解为用户特征矩阵和物品特征矩阵，预测缺失评分。

**图像压缩**：

将图像矩阵做截断 SVD，只存储前 $k$ 个奇异值和对应的奇异向量，大幅减少存储量。

---

## 10. 谱分解

### 10.1 定义

若 $A$ 是可对角化的 $n$ 阶方阵，$A = PDP^{-1}$，则：

$$A = \sum_{i=1}^{n} \lambda_i \mathbf{p}_i \mathbf{q}_i^T$$

其中 $\mathbf{p}_i$ 是 $P$ 的第 $i$ 列（右特征向量），$\mathbf{q}_i^T$ 是 $P^{-1}$ 的第 $i$ 行（左特征向量）。

### 10.2 实对称矩阵的谱分解

$$A = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

其中 $\mathbf{q}_i$ 是标准正交特征向量。每个 $\mathbf{q}_i \mathbf{q}_i^T$ 是秩 1 的正交投影矩阵。

---

## 11. Cholesky 分解

### 11.1 定义

若 $A$ 是正定对称矩阵，则存在唯一的下三角矩阵 $L$（对角元素为正）使得：

$$A = LL^T$$

### 11.2 计算公式

$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$

$$l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk}\right), \quad i > j$$

### 11.3 性质与应用

- 正定矩阵的 Cholesky 分解存在且唯一
- 计算量约为 LU 分解的一半（$n^3/3$ vs $2n^3/3$）
- 广泛用于求解正定线性方程组、蒙特卡罗模拟中的协方差矩阵分解、优化算法中
- 若 $A$ 仅为半正定，$L$ 的对角元素可能为零
