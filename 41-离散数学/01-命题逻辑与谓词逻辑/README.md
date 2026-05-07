# 01-命题逻辑与谓词逻辑

逻辑是离散数学的核心基础，命题逻辑和谓词逻辑构成了形式推理的基本框架，是编译原理、数据库查询优化、人工智能推理引擎的理论基石。

---

## 1. 命题与联结词

### 1.1 命题的定义

**命题**是一个能判断真假的陈述句。判断的结果称为命题的**真值**，取值为真（T/1）或假（F/0）。

- **原子命题**（简单命题）：不能再分解为更简单命题的命题，如"2是偶数"。
- **复合命题**：由原子命题通过联结词组合而成。

注意：感叹句、祈使句、疑问句、悖论（如"这句话是假的"）都不是命题。

### 1.2 五个基本联结词

| 联结词 | 符号 | 名称 | 真值规则 |
|--------|------|------|----------|
| 非 P | $\neg P$ | 否定 | P 为真时 $\neg P$ 为假 |
| P 且 Q | $P \wedge Q$ | 合取 | 仅当 P、Q 均为真时为真 |
| P 或 Q | $P \vee Q$ | 析取 | 仅当 P、Q 均为假时为假 |
| 若 P 则 Q | $P \rightarrow Q$ | 蕴含 | 仅当 P 为真且 Q 为假时为假 |
| P 当且仅当 Q | $P \leftrightarrow Q$ | 等价 | P、Q 真值相同时为真 |

**蕴含的常见误区**：$P \rightarrow Q$ 在 P 为假时恒为真（"假命题蕴含任何命题"），这称为**实质蕴含**。

**常见等价表达**：
- "只要 P，就 Q" $\Leftrightarrow P \rightarrow Q$
- "只有 Q，才 P" $\Leftrightarrow P \rightarrow Q$
- "除非 Q，否则不 P" $\Leftrightarrow P \rightarrow Q$

---

## 2. 命题公式的分类

设 $A$ 为一个命题公式：

| 类型 | 定义 | 例子 |
|------|------|------|
| **重言式**（永真式） | 在所有赋值下均为真 | $P \vee \neg P$ |
| **矛盾式**（永假式） | 在所有赋值下均为假 | $P \wedge \neg P$ |
| **可满足式** | 至少存在一个赋值使其为真 | $P \vee Q$ |

**判定方法**：真值表法。对 $n$ 个命题变元，需要检查 $2^n$ 种赋值。

---

## 3. 等值演算与对偶律

### 3.1 基本等值式

| 名称 | 公式 |
|------|------|
| 双重否定律 | $\neg\neg P \Leftrightarrow P$ |
| 幂等律 | $P \wedge P \Leftrightarrow P$, $P \vee P \Leftrightarrow P$ |
| 交换律 | $P \wedge Q \Leftrightarrow Q \wedge P$, $P \vee Q \Leftrightarrow Q \vee P$ |
| 结合律 | $(P \wedge Q) \wedge R \Leftrightarrow P \wedge (Q \wedge R)$ |
| 分配律 | $P \wedge (Q \vee R) \Leftrightarrow (P \wedge Q) \vee (P \wedge R)$ |
| 德摩根律 | $\neg(P \wedge Q) \Leftrightarrow \neg P \vee \neg Q$ |
| 吸收律 | $P \wedge (P \vee Q) \Leftrightarrow P$ |
| 零律 | $P \wedge 0 \Leftrightarrow 0$, $P \vee 1 \Leftrightarrow 1$ |
| 同一律 | $P \wedge 1 \Leftrightarrow P$, $P \vee 0 \Leftrightarrow P$ |
| 排中律 | $P \vee \neg P \Leftrightarrow 1$ |
| 矛盾律 | $P \wedge \neg P \Leftrightarrow 0$ |

### 3.2 蕴含等值式

$$P \rightarrow Q \Leftrightarrow \neg P \vee Q$$

这是将蕴含转化为析取的关键等价式，是推理过程中的核心工具。

### 3.3 对偶律

将公式中的 $\wedge$ 与 $\vee$ 互换、$1$ 与 $0$ 互换，得到的公式称为**对偶式**，记为 $A^*$。

**对偶定理**：若 $A \Leftrightarrow B$，则 $A^* \Leftrightarrow B^*$。

---

## 4. 范式

### 4.1 基本概念

- **文字**：命题变元或其否定。
- **简单析取式**：若干文字的析取，如 $P \vee \neg Q \vee R$。
- **简单合取式**：若干文字的合取，如 $P \wedge \neg Q \wedge R$。
- **析取范式**（DNF）：若干简单合取式的析取。
- **合取范式**（CNF）：若干简单析取式的合取。

### 4.2 主范式

- **极小项**：包含所有命题变元的简单合取式，每个变元出现且仅出现一次。$n$ 个变元共有 $2^n$ 个极小项，记为 $m_i$。
- **极大项**：包含所有命题变元的简单析取式，记为 $M_i$。

**主析取范式**：所有使公式为真的极小项的析取。每个可满足式都有唯一的主析取范式。

**主合取范式**：所有使公式为假的极大项的合取。

**对应关系**：$m_i \Leftrightarrow \neg M_i$，主析取范式与主合取范式互为对偶。

### 4.3 求范式的步骤

1. 消去 $\rightarrow$ 和 $\leftrightarrow$：用 $\neg P \vee Q$ 替换 $P \rightarrow Q$。
2. 将 $\neg$ 移到文字前：反复使用德摩根律和双重否定律。
3. 利用分配律展开。

---

## 5. 命题逻辑推理规则

**推理**：从前提 $A_1, A_2, \ldots, A_n$ 推出结论 $B$，即 $A_1 \wedge A_2 \wedge \cdots \wedge A_n \Rightarrow B$。

### 5.1 常用推理规则

| 规则 | 形式 | 说明 |
|------|------|------|
| **假言推理**（MP） | $P, P \rightarrow Q \Rightarrow Q$ | 肯定前件 |
| **拒取式**（MT） | $\neg Q, P \rightarrow Q \Rightarrow \neg P$ | 否定后件 |
| **假言三段论**（HS） | $P \rightarrow Q, Q \rightarrow R \Rightarrow P \rightarrow R$ | 链式推理 |
| **析取三段论**（DS） | $P \vee Q, \neg P \Rightarrow Q$ | 排除一个 |
| **构造性二难** | $P \rightarrow Q, R \rightarrow S, P \vee R \Rightarrow Q \vee S$ | 分情况推理 |
| **化简** | $P \wedge Q \Rightarrow P$ | 合取消除 |
| **附加** | $P \Rightarrow P \vee Q$ | 析取引入 |
| **合取** | $P, Q \Rightarrow P \wedge Q$ | 合取构造 |

### 5.2 推理方法

- **直接证明法**：从前提逐步推导结论。
- **CP 规则**（条件证明）：要证 $P \Rightarrow (Q \rightarrow R)$，可将 $P, Q$ 作为前提，证明 $R$。
- **反证法**（归谬法）：假设结论的否定，推出矛盾。

---

## 6. 谓词与量词

### 6.1 谓词的概念

**谓词**是描述个体性质或个体间关系的词。将原子命题分解为"个体"和"谓词"：

- $P(x)$："x 是素数"，其中 $P$ 是一元谓词。
- $L(x, y)$："x 大于 y"，其中 $L$ 是二元谓词。
- $n$ 元谓词涉及 $n$ 个个体。

**个体常元**（a, b, c）代表确定的个体，**个体变元**（x, y, z）代表不确定的个体。

### 6.2 量词

| 量词 | 符号 | 含义 | 例子 |
|------|------|------|------|
| **全称量词** | $\forall x$ | "对所有 x" | $\forall x P(x)$：所有 x 都具有性质 P |
| **存在量词** | $\exists x$ | "存在 x" | $\exists x P(x)$：存在某个 x 具有性质 P |

**量词的辖域**：量词后紧跟的最小子公式。

**约束变元与自由变元**：在量词辖域内出现的同名变元为约束变元，否则为自由变元。

---

## 7. 谓词公式的解释与分类

对谓词公式进行**解释** $I$，需要指定：
1. 非空的个体域 $D$。
2. 为每个常元符号指定 $D$ 中的个体。
3. 为每个谓词符号指定 $D$ 上的关系。
4. 为自由变元指定值。

若公式在所有非空解释下均为真，则为**永真式**；存在至少一个解释使其为真，则为**可满足式**。

---

## 8. 谓词逻辑推理规则

### 8.1 四条核心规则

| 规则 | 形式 | 条件 |
|------|------|------|
| **全称量词消去**（UI） | $\forall x P(x) \Rightarrow P(c)$ | c 为个体域中任意个体 |
| **全称量词引入**（UG） | $P(c) \Rightarrow \forall x P(x)$ | c 为任意个体，不能是特殊选取的 |
| **存在量词消去**（EI） | $\exists x P(x) \Rightarrow P(c)$ | c 为某个使 P 成立的特定个体，c 不在前面出现过 |
| **存在量词引入**（EG） | $P(c) \Rightarrow \exists x P(x)$ | c 为个体域中某个个体 |

### 8.2 使用注意事项

- EI 选出的 c 是"新符号"，不能在之前的推理中使用过。
- UG 不能对 EI 引入的常元使用（因其不具"任意性"）。
- 先使用 EI 再使用 UI，顺序很重要。

---

## 9. 前束范式

**定义**：若谓词公式等价于形如 $Q_1 x_1 Q_2 x_2 \cdots Q_n x_n \, B$ 的公式，其中 $Q_i$ 为量词（$\forall$ 或 $\exists$），$B$ 为不含量词的公式，则称该形式为**前束范式**。

**化为前束范式的步骤**：
1. 消去 $\rightarrow$ 和 $\leftrightarrow$。
2. 将 $\neg$ 内移至文字前（利用德摩根律和量词否定律）。
3. 变元更名使约束变元不重名。
4. 将所有量词提到最前面。

**量词否定律**：
- $\neg \forall x P(x) \Leftrightarrow \exists x \neg P(x)$
- $\neg \exists x P(x) \Leftrightarrow \forall x \neg P(x)$

---

## 10. 典型例题

**例 1**：将"如果明天下雨，我就带伞"转化为命题公式。
设 $P$：明天下雨，$Q$：我带伞，则公式为 $P \rightarrow Q$。

**例 2**：求 $P \rightarrow (Q \rightarrow R)$ 的主析取范式。

解：
$$P \rightarrow (Q \rightarrow R) \Leftrightarrow \neg P \vee (\neg Q \vee R) \Leftrightarrow \neg P \vee \neg Q \vee R$$

这是简单析取式，也是合取范式。展开为极小项：
$$= m_0 \vee m_1 \vee m_2 \vee m_3 \vee m_4 \vee m_5 \vee m_7$$

**例 3**：证明 $\forall x(P(x) \rightarrow Q(x)), \forall x P(x) \Rightarrow \forall x Q(x)$。

证明：
1. $\forall x P(x)$ （前提）
2. $P(a)$ （UI, 1）
3. $\forall x(P(x) \rightarrow Q(x))$ （前提）
4. $P(a) \rightarrow Q(a)$ （UI, 3）
5. $Q(a)$ （MP, 2, 4）
6. $\forall x Q(x)$ （UG, 5）

---

> **408 考研要点**：重点掌握联结词真值表、等值演算、范式求法、推理规则，以及谓词逻辑中量词的消去与引入规则。
