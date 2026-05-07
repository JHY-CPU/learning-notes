# 01-有穷自动机与正则语言

## 1. 语言与字母表的基本概念

**字母表（Alphabet）**：一个非空的有穷集合，通常记为 $\Sigma$。例如 $\Sigma = \{0, 1\}$ 是二进制字母表。

**字符串（String / Word）**：字母表中符号的有穷序列。空字符串记为 $\varepsilon$（或 $\lambda$），长度为 0。

**字符串的长度**：字符串 $w$ 中符号的个数，记为 $|w|$。

**字符串的连接（Concatenation）**：若 $x = a_1 a_2 \cdots a_m$，$y = b_1 b_2 \cdots b_n$，则 $xy = a_1 a_2 \cdots a_m b_1 b_2 \cdots b_n$。

**Kleene 星（Kleene Star）**：$\Sigma^*$ 表示 $\Sigma$ 上所有有穷字符串（包括 $\varepsilon$）的集合。$\Sigma^+ = \Sigma^* \setminus \{\varepsilon\}$。

**语言（Language）**：$\Sigma^*$ 的任意子集称为 $\Sigma$ 上的语言。例如 $L = \{0^n 1^n \mid n \geq 0\}$ 是一个语言。

---

## 2. DFA（确定性有穷自动机）的形式定义

**定义**：DFA 是一个五元组 $M = (Q, \Sigma, \delta, q_0, F)$，其中：

- $Q$：有穷的状态集合
- $\Sigma$：有穷的输入字母表
- $\delta: Q \times \Sigma \to Q$：转移函数
- $q_0 \in Q$：初始状态
- $F \subseteq Q$：接受状态（终结状态）集合

**扩展转移函数 $\hat{\delta}$**：

- $\hat{\delta}(q, \varepsilon) = q$
- $\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a)$，其中 $w \in \Sigma^*$，$a \in \Sigma$

**语言接受**：DFA $M$ 接受的语言为 $L(M) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \in F\}$。

**状态转移图**：用节点表示状态，有向边表示转移，边上的标签为输入符号。初始状态用箭头标记，接受状态用双圈标记。

**状态转移表**：以表格形式列出 $\delta(q, a)$ 对每个 $(q, a)$ 的值。

**例题**：构造 DFA 接受所有以 "01" 结尾的二进制字符串。

- 状态：$q_0$（初始）、$q_1$（刚读到 0）、$q_2$（刚读到 01，接受状态）
- 转移：$\delta(q_0, 0) = q_1$，$\delta(q_0, 1) = q_0$，$\delta(q_1, 0) = q_1$，$\delta(q_1, 1) = q_2$，$\delta(q_2, 0) = q_1$，$\delta(q_2, 1) = q_0$

---

## 3. NFA（非确定性有穷自动机）的形式定义

**定义**：NFA 是五元组 $M = (Q, \Sigma, \delta, q_0, F)$，其中转移函数为 $\delta: Q \times \Sigma \to 2^Q$（返回状态集合）。

**关键区别**：DFA 对每个状态和输入符号恰好有一个后继状态；NFA 可以有零个、一个或多个后继状态。

**语言接受**：NFA 接受字符串 $w$，当且仅当存在**至少一条**计算路径使得 $\hat{\delta}(q_0, w) \cap F \neq \emptyset$。

**定理**：NFA 与 DFA 是等价的——对于每个 NFA，存在一个 DFA 接受相同的语言。

---

## 4. NFA 到 DFA 的转换（子集构造法）

**子集构造法（Subset Construction / Powerset Construction）**：

给定 NFA $N = (Q_N, \Sigma, \delta_N, q_0, F_N)$，构造等价的 DFA $D = (Q_D, \Sigma, \delta_D, q_{0D}, F_D)$：

- $Q_D = 2^{Q_N}$（$Q_N$ 的幂集，最多 $2^{|Q_N|}$ 个状态）
- $q_{0D} = \{q_0\}$
- $F_D = \{S \subseteq Q_N \mid S \cap F_N \neq \emptyset\}$
- $\delta_D(S, a) = \bigcup_{q \in S} \delta_N(q, a)$

**证明思路**：对 $|w|$ 归纳证明 $\hat{\delta}_D(\{q_0\}, w) = \hat{\delta}_N(q_0, w)$。

**最坏情况**：子集构造法产生的 DFA 状态数可达 $2^{|Q_N|}$，这是紧的——存在 NFA 使得等价 DFA 必须有指数多个状态。

---

## 5. ε-NFA 及其等价性

**定义**：ε-NFA 在 NFA 基础上增加了 $\varepsilon$-转移：$\delta: Q \times (\Sigma \cup \{\varepsilon\}) \to 2^Q$。

**ε-闭包**：$\text{ECLOSE}(q)$ 是从状态 $q$ 出发仅通过 $\varepsilon$-转移可以到达的所有状态的集合（包括 $q$ 自身）。

$$\text{ECLOSE}(q) = \{p \mid \text{存在从 } q \text{ 到 } p \text{ 的 } \varepsilon\text{-转移路径}\}$$

**等价性**：ε-NFA $\Leftrightarrow$ NFA $\Leftrightarrow$ DFA。三者描述的语言类完全相同，即**正则语言**。

**ε-NFA 转 DFA**：将子集构造法中的转移改为：

$$\delta_D(S, a) = \bigcup_{q \in S} \text{ECLOSE}(\delta_N(q, a))$$

---

## 6. 正则表达式的定义

**定义**：在字母表 $\Sigma$ 上的正则表达式递归定义如下：

1. **基础**：$\emptyset$、$\varepsilon$、$a$（$a \in \Sigma$）都是正则表达式
2. **归纳**：若 $R$ 和 $S$ 是正则表达式，则 $(R+S)$、$(RS)$、$(R^*)$ 也是正则表达式
3. **限制**：仅由以上规则产生

**正则表达式描述的语言**：每个正则表达式 $R$ 描述一个语言 $L(R)$。

---

## 7. 正则表达式与 FA 的等价性

**定理**：语言 $L$ 是正则语言，当且仅当存在正则表达式 $R$ 使得 $L = L(R)$。

**正则表达式 $\to$ NFA**（Thompson 构造法）：

- 基础情况：为 $a$、$\varepsilon$、$\emptyset$ 构造对应的 NFA
- 归纳情况：利用 $\varepsilon$-转移将子 NFA 连接起来（并、连接、Kleene 星）

**NFA $\to$ 正则表达式**（状态消除法）：

- 引入新的初始状态和接受状态
- 逐步消除中间状态，同时更新转移上的正则表达式标签
- 最终只剩两个状态之间的正则表达式即为所求

---

## 8. 正则语言的封闭性

**定理**：正则语言类对以下运算是封闭的：

| 运算 | 证明方法 |
|------|---------|
| 并集 $L_1 \cup L_2$ | ε-NFA 并行连接 |
| 交集 $L_1 \cap L_2$ | DFA 乘积构造 |
| 补集 $\overline{L}$ | DFA 接受状态取反 |
| 连接 $L_1 \cdot L_2$ | ε-NFA 串联连接 |
| Kleene 星 $L^*$ | ε-NFA 循环连接 |
| 反转 $L^R$ | NFA 反转所有边 |
| 同态 | 替换每个符号为字符串 |
| 逆同态 | 替换输入符号 |

**乘积构造法**（用于交集）：给定 DFA $A_1 = (Q_1, \Sigma, \delta_1, q_{01}, F_1)$ 和 $A_2 = (Q_2, \Sigma, \delta_2, q_{02}, F_2)$，构造 $A = (Q_1 \times Q_2, \Sigma, \delta, (q_{01}, q_{02}), F_1 \times F_2)$，其中 $\delta((p, q), a) = (\delta_1(p, a), \delta_2(q, a))$。

---

## 9. 泵引理（Pumping Lemma）

**定理（正则语言的泵引理）**：若 $L$ 是正则语言，则存在泵长度 $p > 0$，使得对任意 $w \in L$ 且 $|w| \geq p$，存在分解 $w = xyz$ 满足：

1. $|y| > 0$（$y$ 非空）
2. $|xy| \leq p$
3. 对所有 $i \geq 0$，$xy^i z \in L$

**证明思路**：利用鸽巢原理。若 $|w| \geq p$（$p$ 为 DFA 状态数），则在处理 $w$ 的前 $p$ 个字符时，DFA 必然访问某个状态两次，形成循环。沿循环重复任意次数仍能被接受。

**应用——证明语言不是正则的**：

1. 假设 $L$ 是正则的，设泵长度为 $p$
2. 选择一个特定的 $w \in L$ 使得 $|w| \geq p$（选择要巧妙）
3. 对所有满足泵引理条件的分解 $w = xyz$
4. 找到某个 $i$ 使得 $xy^i z \notin L$
5. 得出矛盾，故 $L$ 不是正则的

**例题**：证明 $L = \{0^n 1^n \mid n \geq 0\}$ 不是正则语言。

取 $w = 0^p 1^p$。由 $|xy| \leq p$，$y$ 只含 0。取 $i = 2$，则 $xy^2 z = 0^{p+|y|} 1^p \notin L$，矛盾。

---

## 10. Myhill-Nerode 定理与最小 DFA

**右不变等价关系**：等价关系 $\equiv_L$ 定义为：$x \equiv_L y$ 当且仅当对所有 $z \in \Sigma^*$，$xz \in L \iff yz \in L$。

**定理（Myhill-Nerode）**：以下三个条件等价：

1. $L$ 是正则语言
2. $\equiv_L$ 的等价类个数有限
3. 存在接受 $L$ 的 DFA，其状态数等于 $\equiv_L$ 的等价类个数

**推论**：最小 DFA 的状态数恰好等于 $\equiv_L$ 的等价类个数。

---

## 11. DFA 最小化算法

**Hopcroft 最小化算法**（分割细化法）：

1. 将所有状态分为两组：接受状态 $F$ 和非接受状态 $Q \setminus F$
2. 不断细化分割：若同一组中的两个状态 $p, q$ 对某个输入符号 $a$ 转移到不同组，则将它们分到不同组
3. 重复直到分割不再变化
4. 每个最终组对应最小 DFA 中的一个状态

**等价表述**：

- 先去除不可达状态
- 将等价状态合并（两个状态等价意味着对所有输入串有相同的接受性）

**时间复杂度**：$O(|\Sigma| \cdot |Q| \log |Q|)$（使用适当的集合数据结构）。

**例题**：对给定 DFA，先标出可区分的状态对（接受 vs 非接受），再逐步扩展，最后合并不可区分的状态。
