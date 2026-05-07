# 03-图灵机与可计算性

## 1. 图灵机的形式定义

**定义**：图灵机（Turing Machine, TM）是七元组 $M = (Q, \Sigma, \Gamma, \delta, q_0, q_{accept}, q_{reject})$，其中：

- $Q$：有穷状态集
- $\Sigma$：输入字母表（不包含空白符 $\sqcup$）
- $\Gamma$：带子字母表，$\Sigma \subseteq \Gamma$，$\sqcup \in \Gamma \setminus \Sigma$
- $\delta: Q \times \Gamma \to Q \times \Gamma \times \{L, R\}$：转移函数
- $q_0 \in Q$：初始状态
- $q_{accept} \in Q$：接受状态
- $q_{reject} \in Q$：拒绝状态（$q_{accept} \neq q_{reject}$）

**运行**：图灵机有一条双向无限长的带子，读写头初始位置在最左端。输入串写在带子上，其余位置为 $\sqcup$。

每一步的转移 $\delta(q, a) = (q', b, D)$ 表示：在状态 $q$ 读到符号 $a$ 时，改写为 $b$，转移到状态 $q'$，读写头向方向 $D$（$L$ 或 $R$）移动一格。

**语言接受**：$L(M) = \{w \in \Sigma^* \mid M \text{ 在输入 } w \text{ 上停机在 } q_{accept}\}$。

**注意**：图灵机可能在某些输入上不停机（无限循环），这是与 DFA/NFA/PDA 的根本区别。

---

## 2. 图灵机的变体

### 多带图灵机

**定义**：有 $k$ 条带子和 $k$ 个读写头的图灵机。转移函数：

$$\delta: Q \times \Gamma^k \to Q \times \Gamma^k \times \{L, R\}^k$$

**定理**：多带图灵机与单带图灵机等价。

**证明思路**：将多带模拟为单带——用一条带子存储所有带子的内容（交替存放），用标记符号记录每个读写头的位置。模拟一步多带操作需要在单带上多次扫描。

### 非确定性图灵机

**定义**：转移函数为 $\delta: Q \times \Gamma \to 2^{Q \times \Gamma \times \{L, R\}}$。

**接受定义**：非确定性 TM 接受输入 $w$，如果存在**至少一条**计算路径导致 $q_{accept}$。

**定理**：非确定性图灵机与确定性图灵机等价。

**证明思路**：DTM 模拟 NTM 的所有可能计算路径（使用多带——一条记录当前配置，一条作为队列进行 BFS 搜索）。

**注意**：NTM 与 DTM 的等价仅在语言识别层面上成立。在时间复杂度上，NTM 可能更强大（即 NP 问题）。

### 枚举器（Enumerator）

**定义**：带有打印机输出的图灵机。它不接受语言，而是枚举（打印出）一个语言的所有字符串。

**定理**：语言 $L$ 是递归可枚举的，当且仅当存在枚举器枚举 $L$。

---

## 3. 图灵机与算法的关系

**Church-Turing 论题**：任何在直觉上"有效可计算"的函数，都可以由图灵机计算。

这个论题**不是**数学定理，而是一个关于"算法"概念的精确化猜想。所有已知的计算模型（$\lambda$-演算、递归函数、Post 系统、RAM 模型等）都被证明与图灵机等价，这为 Church-Turing 论题提供了强有力的支持。

**推论**：图灵机可以模拟任何实际计算机程序。任何算法都可以被写成图灵机。

---

## 4. 可判定语言与不可判定语言

**可判定语言（Decidable / Recursive）**：存在图灵机 $M$ 使得对所有输入 $w$，$M$ 都停机，且 $w \in L \iff M$ 接受 $w$。

**图灵可识别语言（Turing-recognizable / Recursively Enumerable）**：存在图灵机 $M$ 使得 $L(M) = L$（$M$ 对 $w \in L$ 停机接受，对 $w \notin L$ 可能拒绝或不停机）。

**关系**：

- 可判定 $\subset$ 图灵可识别 $\subset$ 所有语言
- $L$ 可判定 $\iff$ $L$ 和 $\overline{L}$ 都是图灵可识别的
- 存在语言不是图灵可识别的（由可枚举性论证：图灵机编码可数，语言不可数）

---

## 5. 停机问题的不可判定性

**定理（图灵, 1936）**：语言 $A_{TM} = \{\langle M, w \rangle \mid M \text{ 是 TM 且 } M \text{ 接受 } w\}$ 是图灵可识别但不可判定的。

**证明（对角化）**：

$A_{TM}$ 的图灵可识别性是显然的——通用图灵机可以直接模拟 $M$ 在 $w$ 上的运行。

不可判定性的证明使用对角化论证：

1. 假设 $A_{TM}$ 可判定，存在 TM $H$ 使得 $H(\langle M, w \rangle)$ 在 $M$ 接受 $w$ 时返回 accept，否则返回 reject
2. 构造 TM $D$：$D(\langle M \rangle) = $ 若 $H(\langle M, \langle M \rangle \rangle) = \text{accept}$ 则 reject，否则 accept
3. 考虑 $D(\langle D \rangle)$：
   - 若 $D$ 接受 $\langle D \rangle$，则 $H$ 接受 $\langle D, \langle D \rangle \rangle$，则 $D$ 拒绝 $\langle D \rangle$——矛盾
   - 若 $D$ 拒绝 $\langle D \rangle$，则 $H$ 拒绝 $\langle D, \langle D \rangle \rangle$，则 $D$ 接受 $\langle D \rangle$——矛盾
4. 故假设错误，$A_{TM}$ 不可判定

**停机问题**：$\text{HALT}_{TM} = \{\langle M, w \rangle \mid M \text{ 在 } w \text{ 上停机}\}$ 也是不可判定的。

---

## 6. Rice 定理

**定理（Rice）**：设 $\mathcal{P}$ 是图灵可识别语言的非平凡性质（即 $\mathcal{P} \neq \emptyset$ 且 $\mathcal{P}$ 不包含所有图灵可识别语言），则

$$L_{\mathcal{P}} = \{\langle M \rangle \mid L(M) \in \mathcal{P}\}$$

是不可判定的。

**"非平凡"的含义**：存在图灵机的语言满足该性质，也存在图灵机的语言不满足该性质。

**证明思路**：归约自 $A_{TM}$。不失一般性，假设 $\emptyset \notin \mathcal{P}$，取 $L_0 \in \mathcal{P}$，构造归约 $f(\langle M, w \rangle)$ 输出一个 TM 的编码，该 TM 模拟 $M$ 在 $w$ 上的运行，若接受则模拟 $L_0$ 的识别。

**应用**：Rice 定理可以批量证明许多性质的不可判定性，例如：

- "L(M) 是正则语言"——不可判定
- "L(M) 是空集"——不可判定
- "L(M) 是无穷集"——不可判定
- "L(M) = Σ*"——不可判定

---

## 7. 归约方法

**归约的核心思想**：若问题 $A$ 可归约到问题 $B$（记为 $A \leq_m B$），则"$B$ 可判定 $\Rightarrow$ $A$ 可判定"。等价地，"$A$ 不可判定 $\Rightarrow$ $B$ 不可判定"。

### 映射归约（Mapping Reduction）

**定义**：$A \leq_m B$，如果存在可计算函数 $f: \Sigma^* \to \Sigma^*$，使得对所有 $w$：

$$w \in A \iff f(w) \in B$$

**构造归约的步骤**：

1. 已知 $A$ 不可判定
2. 对任意输入 $w$，构造 $f(w)$（$f$ 必须是图灵可计算的）
3. 证明 $w \in A \iff f(w) \in B$
4. 结论：$B$ 不可判定

### 计算归约

更一般地，$A$ 用 $B$ 作为子程序（oracle）来判定。映射归约是计算归约的特例。

---

## 8. 递归可枚举语言与递归语言

**递归语言（Recursive / Decidable）**：图灵机对所有输入都停机。

**递归可枚举语言（Recursively Enumerable / Turing-recognizable）**：图灵机对语言中的字符串停机接受，对不在语言中的字符串可能不停机。

**基本性质**：

- 递归语言类对并、交、补、连接、Kleene 星封闭
- 递归可枚举语言对并、交、连接、Kleene 星封闭，但对**补集不封闭**
- $L$ 是递归的 $\iff$ $L$ 和 $\overline{L}$ 都是递归可枚举的
- $\overline{A_{TM}}$ 不是递归可枚举的

---

## 9. Post 对应问题

**定义**：给定字符串对的集合 $\{(t_1, b_1), (t_2, b_2), \ldots, (t_k, b_k)\}$，是否存在下标序列 $i_1, i_2, \ldots, i_m$（$m \geq 1$）使得

$$t_{i_1} t_{i_2} \cdots t_{i_m} = b_{i_1} b_{i_2} \cdots b_{i_m}$$

**定理**：Post 对应问题（PCP）是不可判定的。

**证明思路**：归约自 $A_{TM}$。对任意 TM $M$ 和输入 $w$，构造一组字符串对，使得 PCP 有解当且仅当 $M$ 接受 $w$。

**应用**：PCP 的不可判定性用于证明许多形式语言问题的不可判定性，例如 CFL 的等价性问题、CFG 的歧义性问题等。

---

## 10. 可计算函数与部分可计算函数

**可计算函数（Computable / Total Recursive）**：存在对所有输入都停机的图灵机计算该函数。

**部分可计算函数（Partially Computable）**：存在图灵机在输入为 $x$ 时输出 $f(x)$（若 $f(x)$ 有定义），可能不停机（若 $f(x)$ 无定义）。

**通用图灵机**：存在图灵机 $U$，对任意图灵机 $M$ 和输入 $w$，$U(\langle M, w \rangle)$ 模拟 $M$ 在 $w$ 上的运行。这是所有编程语言解释器/编译器的理论基础。

**编码**：图灵机本身可以被编码为字符串（$\langle M \rangle \in \Sigma^*$），使得图灵机可以操作其他图灵机的描述。这是自指和对角化论证的基础。

---

## 11. Gödel 不完备定理的含义

**第一不完备定理**：在任何包含基本算术的一致形式系统中，存在真但不可证明的命题。

**与计算理论的联系**：

- 如果每个真命题都可以被证明（即形式系统是完备的），则真理性可以被算法判定——但 $A_{TM}$ 的不可判定性排除了这种可能
- 不可判定性 $\Rightarrow$ 不完备性：如果系统可以判定自身的一致性，则停机问题可判定，矛盾
- 图灵的不可判定性结果实际上是 Gödel 不完备定理的推广

**核心思想**：任何足够强大的形式系统都面临"自指"带来的限制——这与图灵机对角化论证、Rice 定理中的思想一脉相承。
