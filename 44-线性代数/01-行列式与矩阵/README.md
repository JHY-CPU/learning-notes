# 01 - 行列式与矩阵

## 1. 行列式的定义

### 1.1 排列逆序数法定义

设 $A = (a_{ij})$ 是一个 $n$ 阶方阵，其行列式定义为：

$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}$$

其中 $S_n$ 是 $n$ 个元素的所有排列的集合，$\text{sgn}(\sigma)$ 是排列 $\sigma$ 的符号（偶排列为 $+1$，奇排列为 $-1$）。

**逆序数**：在一个排列 $j_1 j_2 \cdots j_n$ 中，若 $i < k$ 但 $j_i > j_k$，则称 $(j_i, j_k)$ 为一个逆序。排列中逆序的总数称为逆序数，记为 $\tau(j_1 j_2 \cdots j_n)$。

$$\text{sgn}(\sigma) = (-1)^{\tau(\sigma)}$$

### 1.2 递归展开定义（拉普拉斯展开）

对于 $n$ 阶方阵 $A$，按第 $i$ 行展开：

$$\det(A) = \sum_{j=1}^{n} a_{ij} A_{ij}$$

其中 $A_{ij} = (-1)^{i+j} M_{ij}$ 是元素 $a_{ij}$ 的**代数余子式**，$M_{ij}$ 是删去第 $i$ 行和第 $j$ 列后得到的 $(n-1)$ 阶子式。

特别地，二阶和三阶行列式有直接计算公式：

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

$$\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}$$

---

## 2. 行列式的性质

行列式具有以下基本性质，这些性质是简化行列式计算的核心工具。

### 2.1 转置不变性

$$\det(A^T) = \det(A)$$

这意味着**对行成立的性质对列也成立**。

### 2.2 交换两行（列）变号

若 $A'$ 是交换 $A$ 的第 $i$ 行与第 $j$ 行得到的矩阵，则：

$$\det(A') = -\det(A)$$

**推论**：若 $A$ 有两行（列）相同，则 $\det(A) = 0$。

### 2.3 行（列）的倍加不变性

将第 $j$ 行的 $k$ 倍加到第 $i$ 行，行列式的值不变。即初等变换 $r_i \leftarrow r_i + k r_j$ 不改变行列式的值。

这个性质在计算行列式时极为重要，可用于将矩阵化为上三角形式。

### 2.4 数乘性质

若将行列式的某一行（列）乘以常数 $k$，则行列式的值变为原来的 $k$ 倍：

$$\det(\cdots, k \cdot r_i, \cdots) = k \cdot \det(\cdots, r_i, \cdots)$$

**注意**：$\det(kA) = k^n \det(A)$，其中 $n$ 是矩阵的阶数，这是因为每一行都乘了 $k$。

### 2.5 可加性

若某一行可以写成两行之和，则行列式可以拆成两个行列式之和：

$$\det(\cdots, r_i + r_i', \cdots) = \det(\cdots, r_i, \cdots) + \det(\cdots, r_i', \cdots)$$

---

## 3. 行列式按行/列展开

### 3.1 代数余子式展开

按第 $i$ 行展开：$\det(A) = \sum_{j=1}^n a_{ij} A_{ij}$

按第 $j$ 列展开：$\det(A) = \sum_{i=1}^n a_{ij} A_{ij}$

### 3.2 拉普拉斯定理（广义展开）

选取 $k$ 行（$1 \leq k \leq n-1$），行列式等于所有由这 $k$ 行构成的 $k$ 阶子式与其对应的代数余子式的乘积之和。

### 3.3 代数余子式的重要性质

- $\sum_{j=1}^n a_{ij} A_{kj} = 0$（当 $i \neq k$ 时，即不同行元素与另一行代数余子式对应乘积之和为零）
- $\sum_{i=1}^n a_{ij} A_{ik} = 0$（当 $j \neq k$ 时）

这两个性质可写成矩阵形式：$A \cdot \text{adj}(A) = \det(A) \cdot I$，其中 $\text{adj}(A)$ 是 $A$ 的伴随矩阵。

---

## 4. 特殊行列式

### 4.1 三角行列式

上三角或下三角矩阵的行列式等于主对角线上元素的乘积：

$$\det\begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ 0 & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_{nn} \end{pmatrix} = \prod_{i=1}^{n} a_{ii}$$

这是计算行列式最高效的形式，也是高斯消元法的核心目标。

### 4.2 范德蒙德（Vandermonde）行列式

$$V_n = \det\begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\ 1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^{n-1} \end{pmatrix} = \prod_{1 \leq i < j \leq n}(x_j - x_i)$$

当 $x_1, x_2, \ldots, x_n$ 两两不等时，范德蒙德行列式非零。

### 4.3 循环行列式

形如：

$$C_n = \det\begin{pmatrix} a_0 & a_1 & a_2 & \cdots & a_{n-1} \\ a_{n-1} & a_0 & a_1 & \cdots & a_{n-2} \\ a_{n-2} & a_{n-1} & a_0 & \cdots & a_{n-3} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ a_1 & a_2 & a_3 & \cdots & a_0 \end{pmatrix}$$

计算公式为 $C_n = \prod_{j=0}^{n-1} f(\omega_j)$，其中 $f(x) = a_0 + a_1 x + \cdots + a_{n-1} x^{n-1}$，$\omega_j = e^{2\pi i j / n}$ 是 $n$ 次单位根。

### 4.4 爪型行列式

主对角线外只有一行和一列非零的行列式，可通过行变换化为三角形式快速计算。

---

## 5. Cramer 法则

若 $A$ 是 $n$ 阶可逆矩阵，线性方程组 $A\mathbf{x} = \mathbf{b}$ 有唯一解：

$$x_j = \frac{\det(A_j)}{\det(A)}, \quad j = 1, 2, \ldots, n$$

其中 $A_j$ 是将 $A$ 的第 $j$ 列替换为 $\mathbf{b}$ 后得到的矩阵。

**适用条件**：$\det(A) \neq 0$。

**注意**：Cramer 法则在理论上非常重要，但在实际计算中效率较低（$O(n \cdot n!)$），高斯消元法（$O(n^3)$）更为实用。

---

## 6. 矩阵基本运算

### 6.1 加法与数乘

- $(A + B)_{ij} = a_{ij} + b_{ij}$
- $(kA)_{ij} = k \cdot a_{ij}$
- 矩阵加法满足交换律和结合律

### 6.2 矩阵乘法

设 $A$ 是 $m \times n$ 矩阵，$B$ 是 $n \times p$ 矩阵，则 $C = AB$ 是 $m \times p$ 矩阵：

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**重要注意**：
- 矩阵乘法一般**不满足交换律**：$AB \neq BA$
- $AB = 0$ 不能推出 $A = 0$ 或 $B = 0$
- 矩阵乘法满足结合律：$(AB)C = A(BC)$
- 满足分配律：$A(B + C) = AB + AC$

### 6.3 转置

$(A^T)_{ij} = a_{ji}$，性质包括：

- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(kA)^T = kA^T$
- $(AB)^T = B^T A^T$（注意顺序翻转）

---

## 7. 逆矩阵

### 7.1 定义

若存在 $n$ 阶矩阵 $B$ 使得 $AB = BA = I$，则 $A$ 可逆，$B$ 称为 $A$ 的逆矩阵，记为 $A^{-1}$。

**可逆的充要条件**：$\det(A) \neq 0$（即 $A$ 是非奇异矩阵）。

### 7.2 伴随矩阵法

$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

其中 $\text{adj}(A)$ 是 $A$ 的伴随矩阵，其第 $i$ 行第 $j$ 列的元素为 $A_{ji}$（注意转置）。

### 7.3 初等变换法（行化简）

构造增广矩阵 $(A \mid I)$，通过行初等变换将左侧化为 $I$，则右侧即为 $A^{-1}$。

### 7.4 逆矩阵的性质

- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1} A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$
- $(kA)^{-1} = \frac{1}{k} A^{-1}$（$k \neq 0$）

---

## 8. 矩阵的秩

### 8.1 定义

矩阵 $A$ 的**秩**（$\text{rank}(A)$）是 $A$ 中非零子式的最高阶数，也等于行阶梯形中非零行的个数。

等价地，$\text{rank}(A) = \dim(\text{col}(A)) = \dim(\text{row}(A))$。

### 8.2 基本性质

- $0 \leq \text{rank}(A) \leq \min(m, n)$
- $\text{rank}(A) = \text{rank}(A^T)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
- $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$
- **Sylvester 不等式**：$\text{rank}(A) + \text{rank}(B) - n \leq \text{rank}(AB)$

### 8.3 满秩矩阵

- $n$ 阶方阵 $A$ 满秩 $\Leftrightarrow$ $\det(A) \neq 0$ $\Leftrightarrow$ $A$ 可逆

---

## 9. 分块矩阵运算

将矩阵按行和列划分为若干子块，运算规则为：

- **加法**：对应子块相加（划分方式需相同）
- **乘法**：$C_{ij} = \sum_k A_{ik} B_{kj}$（要求 $A$ 的列划分与 $B$ 的行划分一致）
- **转置**：先转大矩阵再转每个子块，且子块之间也转置

### 常用分块形式

- **对角分块**：$\text{diag}(A_1, A_2, \ldots, A_k)$，每个 $A_i$ 独立运算
- **上三角分块**：$\begin{pmatrix} A & B \\ 0 & C \end{pmatrix}$，其行列式为 $\det(A) \cdot \det(C)$

---

## 10. 初等矩阵与 LU 分解

### 10.1 初等矩阵

由单位矩阵经过一次初等变换得到的矩阵称为初等矩阵，有三种类型：

- **$E_{ij}$**：交换单位矩阵的第 $i$ 行与第 $j$ 行
- **$E_i(k)$**：将单位矩阵的第 $i$ 行乘以 $k$（$k \neq 0$）
- **$E_{ij}(k)$**：将单位矩阵第 $j$ 行的 $k$ 倍加到第 $i$ 行

**关键性质**：对矩阵 $A$ 做一次初等行变换，等价于左乘相应的初等矩阵。

### 10.2 LU 分解

若 $A$ 可通过不交换行的高斯消元化为上三角矩阵，则：

$$A = LU$$

其中 $L$ 是单位下三角矩阵（对角线全为 1），$U$ 是上三角矩阵。

**Doolittle 分解**：$L$ 的对角线为 1，$U$ 的元素逐步计算。

**Crout 分解**：$U$ 的对角线为 1。

若需要行交换，则 $PA = LU$，其中 $P$ 是置换矩阵。

**应用**：LU 分解可高效求解线性方程组 $A\mathbf{x} = \mathbf{b}$：

1. $L\mathbf{y} = \mathbf{b}$（前代）
2. $U\mathbf{x} = \mathbf{y}$（回代）

时间复杂度为 $O(n^3)$（分解）+$O(n^2)$（求解），当需要对同一系数矩阵求解多个右端项时特别高效。
