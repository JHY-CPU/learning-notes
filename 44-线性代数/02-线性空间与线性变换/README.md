# 02 - 线性空间与线性变换

## 1. 向量空间的定义与性质

### 1.1 定义

设 $V$ 是一个非空集合，$F$ 是一个数域。若在 $V$ 上定义了**加法**和**数乘**两种运算，且满足以下八条公理，则 $V$ 构成 $F$ 上的**向量空间**（线性空间）：

**加法公理：**
1. **封闭性**：$\forall \alpha, \beta \in V$，$\alpha + \beta \in V$
2. **交换律**：$\alpha + \beta = \beta + \alpha$
3. **结合律**：$(\alpha + \beta) + \gamma = \alpha + (\beta + \gamma)$
4. **零元**：存在 $\mathbf{0} \in V$，使得 $\alpha + \mathbf{0} = \alpha$
5. **负元**：$\forall \alpha \in V$，存在 $-\alpha \in V$，使得 $\alpha + (-\alpha) = \mathbf{0}$

**数乘公理：**
6. **封闭性**：$\forall k \in F, \alpha \in V$，$k\alpha \in V$
7. **分配律**：$k(\alpha + \beta) = k\alpha + k\beta$，$(k + l)\alpha = k\alpha + l\alpha$
8. **结合律**：$(kl)\alpha = k(l\alpha)$
9. **单位元**：$1 \cdot \alpha = \alpha$

### 1.2 常见向量空间

| 空间 | 元素 | 域 |
|------|------|------|
| $\mathbb{R}^n$ | $n$ 维实列向量 | $\mathbb{R}$ |
| $\mathbb{C}^n$ | $n$ 维复列向量 | $\mathbb{C}$ |
| $M_{m \times n}(\mathbb{R})$ | $m \times n$ 实矩阵 | $\mathbb{R}$ |
| $\mathbb{R}[x]_n$ | 次数不超过 $n$ 的实系数多项式 | $\mathbb{R}$ |
| $C[a,b]$ | $[a,b]$ 上的连续函数 | $\mathbb{R}$ |

### 1.3 基本性质

- 零向量 $\mathbf{0}$ 是唯一的
- 每个向量的负向量是唯一的
- $0 \cdot \alpha = \mathbf{0}$，$k \cdot \mathbf{0} = \mathbf{0}$
- $(-1)\alpha = -\alpha$
- $k\alpha = \mathbf{0} \Rightarrow k = 0$ 或 $\alpha = \mathbf{0}$

---

## 2. 线性相关与线性无关

### 2.1 线性组合

向量 $\beta$ 是向量组 $\alpha_1, \alpha_2, \ldots, \alpha_s$ 的**线性组合**，若存在标量 $k_1, k_2, \ldots, k_s \in F$ 使得：

$$\beta = k_1 \alpha_1 + k_2 \alpha_2 + \cdots + k_s \alpha_s$$

向量组 $\alpha_1, \ldots, \alpha_s$ **张成**（span）的子空间：

$$\text{span}(\alpha_1, \ldots, \alpha_s) = \{k_1 \alpha_1 + \cdots + k_s \alpha_s \mid k_i \in F\}$$

### 2.2 线性相关

若存在**不全为零**的标量 $k_1, \ldots, k_s$ 使得 $k_1 \alpha_1 + \cdots + k_s \alpha_s = \mathbf{0}$，则称向量组 $\alpha_1, \ldots, \alpha_s$ **线性相关**。

**等价刻画**：向量组线性相关 $\Leftrightarrow$ 其中至少有一个向量可由其余向量线性表示。

### 2.3 线性无关

若 $k_1 \alpha_1 + \cdots + k_s \alpha_s = \mathbf{0}$ 仅当 $k_1 = k_2 = \cdots = k_s = 0$ 时成立，则称向量组**线性无关**。

### 2.4 重要判别准则

- 含零向量的向量组必线性相关
- 单个非零向量线性无关
- 两个向量线性相关 $\Leftrightarrow$ 它们成比例（平行）
- 部分组线性相关 $\Rightarrow$ 整体组线性相关
- 整体组线性无关 $\Rightarrow$ 任意部分组线性无关
- 向量个数 $s >$ 向量维数 $n$ 时，必线性相关

---

## 3. 基与维数

### 3.1 基的定义

向量空间 $V$ 中一组向量 $\{\alpha_1, \alpha_2, \ldots, \alpha_n\}$ 若满足：

1. $\alpha_1, \ldots, \alpha_n$ **线性无关**
2. $V = \text{span}(\alpha_1, \ldots, \alpha_n)$

则称 $\{\alpha_1, \ldots, \alpha_n\}$ 是 $V$ 的一组**基**。

### 3.2 维数

基中向量的个数称为 $V$ 的**维数**，记为 $\dim(V)$。

**关键性质**：
- $V$ 中任意线性无关组的向量个数不超过 $\dim(V)$
- $V$ 中任意 $n$ 个线性无关向量（$n = \dim(V)$）必为一组基
- $V$ 中任意一组张成 $V$ 的向量至少包含 $n$ 个向量，且恰好 $n$ 个时必为基

### 3.3 标准基

$\mathbb{R}^n$ 的标准基：$\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$，其中 $\mathbf{e}_i$ 的第 $i$ 个分量为 1，其余为 0。

---

## 4. 坐标与坐标变换

### 4.1 坐标

设 $\{\alpha_1, \ldots, \alpha_n\}$ 是 $V$ 的一组基，任意向量 $\beta \in V$ 可唯一表示为：

$$\beta = x_1 \alpha_1 + x_2 \alpha_2 + \cdots + x_n \alpha_n$$

向量 $(x_1, x_2, \ldots, x_n)^T$ 称为 $\beta$ 在基 $\{\alpha_1, \ldots, \alpha_n\}$ 下的**坐标**，记为 $[\beta]_\alpha$。

### 4.2 基变换与过渡矩阵

设 $\alpha = \{\alpha_1, \ldots, \alpha_n\}$ 和 $\beta = \{\beta_1, \ldots, \beta_n\}$ 是 $V$ 的两组基，则存在唯一的可逆矩阵 $P$（**过渡矩阵**）使得：

$$[\beta_1, \ldots, \beta_n] = [\alpha_1, \ldots, \alpha_n] \cdot P$$

### 4.3 坐标变换公式

若 $[\beta]_\alpha = P \cdot [\beta]_\beta$，即：

$$[\beta]_\beta = P^{-1} [\beta]_\alpha$$

### 4.4 例题

设 $\mathbb{R}^2$ 的两组基为 $\alpha = \{(1,0)^T, (0,1)^T\}$（标准基）和 $\beta = \{(1,1)^T, (1,-1)^T\}$。

过渡矩阵 $P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$，则 $P^{-1} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$。

若 $\gamma$ 在标准基下坐标为 $(3, 1)^T$，则在 $\beta$ 基下的坐标为 $P^{-1}(3,1)^T = (2, 1)^T$。

---

## 5. 子空间及其交与和

### 5.1 子空间的定义

$V$ 的非空子集 $W$ 若对加法和数乘封闭，则 $W$ 是 $V$ 的**子空间**，记为 $W \leq V$。

**判定定理**：$W \leq V \Leftrightarrow$ $W$ 非空且 $\forall \alpha, \beta \in W, k \in F$：$\alpha + \beta \in W$，$k\alpha \in W$。

### 5.2 生成子空间

$$\text{span}(\alpha_1, \ldots, \alpha_s) = \{k_1 \alpha_1 + \cdots + k_s \alpha_s \mid k_i \in F\}$$

是包含 $\alpha_1, \ldots, \alpha_s$ 的最小子空间。

### 5.3 子空间的交

$$W_1 \cap W_2 = \{\alpha \in V \mid \alpha \in W_1 \text{ 且 } \alpha \in W_2\}$$

$W_1 \cap W_2$ 是子空间。

### 5.4 子空间的和

$$W_1 + W_2 = \{\alpha_1 + \alpha_2 \mid \alpha_1 \in W_1, \alpha_2 \in W_2\}$$

$W_1 + W_2$ 是包含 $W_1$ 和 $W_2$ 的最小子空间。

### 5.5 直和

若 $W_1 \cap W_2 = \{\mathbf{0}\}$，则 $W_1 + W_2$ 称为**直和**，记为 $W_1 \oplus W_2$。

**直和的等价条件**：
- $W_1 \cap W_2 = \{\mathbf{0}\}$
- $\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2)$
- $W_1 + W_2$ 中的零向量表示唯一
- $W_1$ 的基与 $W_2$ 的基合并后仍线性无关

---

## 6. 维数公式

### 6.1 核心公式

对于有限维向量空间 $V$ 的两个子空间 $W_1, W_2$：

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$$

### 6.2 推论

- $\dim(W_1 + W_2) \leq \dim(W_1) + \dim(W_2)$
- 若 $W_1, W_2 \leq V$ 且 $\dim(W_1) + \dim(W_2) > \dim(V)$，则 $W_1 \cap W_2 \neq \{\mathbf{0}\}$

---

## 7. 线性变换的定义与矩阵表示

### 7.1 定义

映射 $T: V \to W$ 若满足：

$$T(\alpha + \beta) = T(\alpha) + T(\beta), \quad T(k\alpha) = kT(\alpha)$$

则称 $T$ 为**线性变换**（线性映射）。

**基本性质**：
- $T(\mathbf{0}) = \mathbf{0}$
- $T(-\alpha) = -T(\alpha)$
- $T(k_1 \alpha_1 + \cdots + k_s \alpha_s) = k_1 T(\alpha_1) + \cdots + k_s T(\alpha_s)$

### 7.2 矩阵表示

设 $\{\alpha_1, \ldots, \alpha_n\}$ 是 $V$ 的基，$\{\beta_1, \ldots, \beta_m\}$ 是 $W$ 的基。线性变换 $T: V \to W$ 在这两组基下的**矩阵表示**为 $A = (a_{ij})_{m \times n}$，其中：

$$T(\alpha_j) = \sum_{i=1}^{m} a_{ij} \beta_i, \quad j = 1, \ldots, n$$

对于 $\gamma \in V$，若 $[\gamma]_\alpha = \mathbf{x}$，则 $[T(\gamma)]_\beta = A\mathbf{x}$。

### 7.3 基变换下的矩阵关系

若 $V$ 的基变换矩阵为 $P$，$W$ 的基变换矩阵为 $Q$，则 $T$ 在新基下的矩阵为：

$$A' = Q^{-1} A P$$

若 $V = W$ 且使用同一组基变换 $P$，则 $A' = P^{-1} A P$（**相似变换**）。

---

## 8. 线性变换的核与像

### 8.1 核（Kernel）

$$\ker(T) = \{\alpha \in V \mid T(\alpha) = \mathbf{0}\}$$

$\ker(T)$ 是 $V$ 的子空间。$\dim(\ker(T))$ 称为 $T$ 的**零度**（nullity）。

### 8.2 像（Image）

$$\text{im}(T) = \{T(\alpha) \mid \alpha \in V\}$$

$\text{im}(T)$ 是 $W$ 的子空间。$\dim(\text{im}(T))$ 称为 $T$ 的**秩**（rank）。

### 8.3 秩-零度定理

$$\dim(V) = \text{rank}(T) + \text{nullity}(T)$$

即：$\dim(V) = \dim(\text{im}(T)) + \dim(\ker(T))$

**矩阵形式**：对于 $m \times n$ 矩阵 $A$：

$$n = \text{rank}(A) + \dim(\ker(A))$$

### 8.4 单射与满射

- $T$ 是单射 $\Leftrightarrow$ $\ker(T) = \{\mathbf{0}\}$ $\Leftrightarrow$ $\text{nullity}(T) = 0$
- $T$ 是满射 $\Leftrightarrow$ $\text{im}(T) = W$ $\Leftrightarrow$ $\text{rank}(T) = \dim(W)$
- $T$ 是双射（同构）$\Leftrightarrow$ $\text{rank}(T) = \dim(V) = \dim(W)$

---

## 9. 不变子空间

### 9.1 定义

$W$ 是 $V$ 的子空间，线性变换 $T: V \to V$。若 $T(W) \subseteq W$，则称 $W$ 是 $T$ 的**不变子空间**。

### 9.2 常见不变子空间

- $\{\mathbf{0}\}$ 和 $V$ 本身（平凡不变子空间）
- $\ker(T)$ 和 $\text{im}(T)$ 都是 $T$ 的不变子空间
- $T$ 的特征子空间 $V_\lambda = \{\alpha \in V \mid T(\alpha) = \lambda \alpha\}$ 是不变子空间

### 9.3 与对角化的关系

$T$ 可对角化 $\Leftrightarrow$ $V$ 可以分解为 $T$ 的特征子空间的直和：

$$V = V_{\lambda_1} \oplus V_{\lambda_2} \oplus \cdots \oplus V_{\lambda_k}$$

### 9.4 循环子空间

对 $\alpha \in V$，由 $\{\alpha, T(\alpha), T^2(\alpha), \ldots\}$ 生成的子空间是 $T$ 的不变子空间，称为由 $\alpha$ 生成的**循环子空间**。
