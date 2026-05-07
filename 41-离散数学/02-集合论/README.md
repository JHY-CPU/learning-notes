# 02-集合论

集合论是现代数学的基础语言，离散数学中的关系、函数、图、代数结构等概念均建立在集合论之上。本章系统介绍集合的基本理论，为后续各章奠定基础。

---

## 1. 集合的基本概念与表示

### 1.1 集合的定义

**集合**是由一些确定的、互不相同的对象汇集而成的整体。集合中的对象称为**元素**。

- $a \in A$：a 属于集合 A
- $a \notin A$：a 不属于集合 A

集合中元素的三个特性：**确定性**、**互异性**、**无序性**。

### 1.2 集合的表示方法

**枚举法**（列举法）：$A = \{1, 2, 3, 4\}$，也可以用省略号如 $B = \{2, 4, 6, \ldots\}$。

**描述法**（谓词法）：$A = \{x \mid P(x)\}$，表示所有满足性质 P 的对象构成的集合。例如：
$$E = \{x \mid x \text{ 是偶数}\}$$

**特殊集合**：
- $\emptyset$（空集）：不包含任何元素的集合
- $\mathbb{N}$：自然数集 $\{0, 1, 2, \ldots\}$
- $\mathbb{Z}$：整数集
- $\mathbb{Q}$：有理数集
- $\mathbb{R}$：实数集
- $\mathbb{C}$：复数集

### 1.3 子集与包含

$A \subseteq B$（A 是 B 的子集）当且仅当 $\forall x(x \in A \rightarrow x \in B)$。

**真子集**：$A \subset B$ 当且仅当 $A \subseteq B \wedge A \neq B$。

**集合相等**：$A = B \Leftrightarrow A \subseteq B \wedge B \subseteq A$（外延公理）。

**重要结论**：
- $\emptyset \subseteq A$ 对任意集合 A 成立
- 含 $n$ 个元素的集合有 $2^n$ 个子集

---

## 2. 集合运算

### 2.1 基本运算

| 运算 | 符号 | 定义 |
|------|------|------|
| **并** | $A \cup B$ | $\{x \mid x \in A \vee x \in B\}$ |
| **交** | $A \cap B$ | $\{x \mid x \in A \wedge x \in B\}$ |
| **差** | $A - B$（或 $A \setminus B$） | $\{x \mid x \in A \wedge x \notin B\}$ |
| **补** | $\bar{A}$（相对于全集 U） | $\{x \mid x \in U \wedge x \notin A\}$ |
| **对称差** | $A \oplus B$ | $(A - B) \cup (B - A) = (A \cup B) - (A \cap B)$ |

若 $A \cap B = \emptyset$，称 A 与 B **不相交**（或**互斥**）。

### 2.2 运算律

| 名称 | 公式 |
|------|------|
| 交换律 | $A \cup B = B \cup A$, $A \cap B = B \cap A$ |
| 结合律 | $(A \cup B) \cup C = A \cup (B \cup C)$ |
| 分配律 | $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ |
| 幂等律 | $A \cup A = A$, $A \cap A = A$ |
| 吸收律 | $A \cup (A \cap B) = A$ |
| 德摩根律 | $\overline{A \cup B} = \bar{A} \cap \bar{B}$, $\overline{A \cap B} = \bar{A} \cup \bar{B}$ |
| 对合律 | $\overline{\bar{A}} = A$ |
| 零律 | $A \cup U = U$, $A \cap \emptyset = \emptyset$ |
| 同一律 | $A \cup \emptyset = A$, $A \cap U = A$ |
| 排中律 | $A \cup \bar{A} = U$ |
| 矛盾律 | $A \cap \bar{A} = \emptyset$ |

---

## 3. 幂集与笛卡尔积

### 3.1 幂集

集合 $A$ 的**幂集** $\mathcal{P}(A)$ 是 $A$ 的所有子集构成的集合：

$$\mathcal{P}(A) = \{S \mid S \subseteq A\}$$

若 $|A| = n$，则 $|\mathcal{P}(A)| = 2^n$。幂集的基数大小是组合计数中的重要结论。

### 3.2 笛卡尔积

两个集合的**笛卡尔积**（直积）定义为：

$$A \times B = \{(a, b) \mid a \in A \wedge b \in B\}$$

若 $|A| = m$，$|B| = n$，则 $|A \times B| = m \cdot n$。

**注意**：笛卡尔积不满足交换律，一般 $A \times B \neq B \times A$。也不满足结合律。

$n$ 个集合的笛卡尔积：
$$A_1 \times A_2 \times \cdots \times A_n = \{(a_1, a_2, \ldots, a_n) \mid a_i \in A_i\}$$

---

## 4. 集合恒等式的证明

### 4.1 常用证明方法

**包含排斥法**（双包含证明）：要证 $A = B$，需分别证明 $A \subseteq B$ 和 $B \subseteq A$。

**元素分析法**：取任意元素 $x$，利用定义分析 $x \in A$ 与 $x \in B$ 的等价关系。

**集合代数法**：直接利用已知恒等式进行推导。

### 4.2 例题

**例**：证明 $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$（分配律）。

**证明**（元素分析法）：

$(\subseteq)$：设 $x \in A \cap (B \cup C)$，则 $x \in A$ 且 $x \in B \cup C$。
- 若 $x \in B$，则 $x \in A \cap B$，故 $x \in (A \cap B) \cup (A \cap C)$。
- 若 $x \in C$，则 $x \in A \cap C$，故 $x \in (A \cap B) \cup (A \cap C)$。

$(\supseteq)$：设 $x \in (A \cap B) \cup (A \cap C)$。
- 若 $x \in A \cap B$，则 $x \in A$ 且 $x \in B$，故 $x \in B \cup C$，从而 $x \in A \cap (B \cup C)$。
- 若 $x \in A \cap C$，同理可证。

---

## 5. 有穷集合与无穷集合的基数

### 5.1 等势

集合 $A$ 与 $B$ **等势**（$|A| = |B|$），如果存在从 $A$ 到 $B$ 的双射函数。

### 5.2 有限集合

有限集合的基数就是其元素个数。对有限集合 $A, B$：

$$|A \cup B| = |A| + |B| - |A \cap B|$$

### 5.3 无穷集合

无穷集合不能与任何自然数集 $\{0, 1, \ldots, n-1\}$ 等势。

**Dedekind 无穷**：若集合与其真子集等势，则为无穷集。

---

## 6. 可数集与不可数集

### 6.1 可数集

**定义**：能与自然数集 $\mathbb{N}$ 建立双射的集合称为**可数集**（可列集）。

**常见可数集**：
- $\mathbb{N}$（自然数集）
- $\mathbb{Z}$（整数集）：可排列为 $0, 1, -1, 2, -2, \ldots$
- $\mathbb{Q}$（有理数集）：可用 Cantor 对角线法证明

**可数集的性质**：
- 可数集的子集仍为可数集（或有穷集）
- 有限个可数集的并集仍为可数集
- 可数个可数集的并集仍为可数集

### 6.2 不可数集

**定理**（Cantor）：实数集 $\mathbb{R}$ 是不可数的。

**证明思路**（对角线法）：假设 $(0,1)$ 区间上的实数可排列为 $r_1, r_2, \ldots$，将每个 $r_i$ 写成十进制小数。构造一个新的实数 $s$，使得 $s$ 的第 $i$ 位与 $r_i$ 的第 $i$ 位不同，则 $s$ 不在列表中，矛盾。

**基数层次**：
$$|\mathbb{N}| = \aleph_0 < |\mathbb{R}| = \mathfrak{c} = 2^{\aleph_0}$$

---

## 7. 鸽巢原理（抽屉原理）

### 7.1 基本形式

**鸽巢原理**：若 $n+1$ 个物体放入 $n$ 个盒子，则至少有一个盒子包含两个或以上的物体。

更一般地：若 $N$ 个物体放入 $k$ 个盒子，则至少有一个盒子包含 $\lceil N/k \rceil$ 个或以上的物体。

### 7.2 应用

**例 1**：在 367 人中，至少有两人生日相同（366 个可能的生日，367 人）。

**例 2**：从 $\{1, 2, \ldots, 200\}$ 中任取 101 个数，必有两个数 $a, b$ 使得 $a \mid b$。

**证明思路**：将 $1, \ldots, 200$ 按最大奇因子分组，共 100 组。取 101 个数，必有两个在同一组，即 $a = 2^i \cdot m$，$b = 2^j \cdot m$（$m$ 为奇数），其中较小者整除较大者。

**例 3**：任意 6 人中，必有 3 人互相认识或互不认识。

---

## 8. 容斥原理

### 8.1 两个集合

$$|A \cup B| = |A| + |B| - |A \cap B|$$

### 8.2 一般形式

设 $A_1, A_2, \ldots, A_n$ 为 $n$ 个集合：

$$\left|\bigcup_{i=1}^n A_i\right| = \sum_{i}|A_i| - \sum_{i<j}|A_i \cap A_j| + \sum_{i<j<k}|A_i \cap A_j \cap A_k| - \cdots + (-1)^{n+1}|A_1 \cap A_2 \cap \cdots \cap A_n|$$

### 8.3 推论：错排数

$n$ 个元素的全排列中，没有一个元素在原来位置上的排列数（**错排数**）：

$$D_n = n!\left(1 - \frac{1}{1!} + \frac{1}{2!} - \frac{1}{3!} + \cdots + (-1)^n \frac{1}{n!}\right)$$

特别地：$D_1 = 0$，$D_2 = 1$，$D_3 = 2$，$D_4 = 9$。

**证明思路**：设 $A_i$ 为第 $i$ 个元素在原位的排列集合，利用容斥原理计算 $|\overline{A_1} \cap \overline{A_2} \cap \cdots \cap \overline{A_n}|$。

### 8.4 Euler 函数

$$\varphi(n) = n \prod_{p \mid n}\left(1 - \frac{1}{p}\right)$$

其中 $p$ 取遍 $n$ 的所有素因子。$\varphi(n)$ 给出 $1$ 到 $n$ 中与 $n$ 互素的整数个数。

---

> **408 考研要点**：集合运算及其恒等式、容斥原理、鸽巢原理的应用是常考内容。注意集合论与命题逻辑在代数结构上的对应关系。
