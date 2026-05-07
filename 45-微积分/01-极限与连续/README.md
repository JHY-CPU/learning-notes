# 01-极限与连续

> 极限是微积分的基石，连续性是函数良好行为的基本要求。本章从严格定义出发，建立极限理论的完整框架。

## 1. 数列极限的ε-N定义

### 1.1 直观理解

数列 $\{a_n\}$ 当 $n$ 无限增大时趋向某个常数 $A$，记为 $\lim_{n\to\infty}a_n=A$。直观上就是"项数足够大时，数列的项可以任意接近 $A$"。

### 1.2 严格定义

$\forall\varepsilon>0$，$\exists N\in\mathbb{N}^*$，当 $n>N$ 时，$|a_n-A|<\varepsilon$。

要点解读：
- $\varepsilon$ 刻画"任意接近"的程度，是先给定的任意正数
- $N$ 依赖于 $\varepsilon$，通常 $\varepsilon$ 越小，需要的 $N$ 越大
- $|a_n-A|<\varepsilon$ 表示 $a_n$ 落在以 $A$ 为中心、$\varepsilon$ 为半径的邻域内
- 数列极限如果存在则唯一

### 1.3 常用证明技巧

- **直接法**：从 $|a_n-A|<\varepsilon$ 出发，反解 $n$，找到 $N$
- **放大法**：将 $|a_n-A|$ 放大为较简单的表达式（通常放大为 $\frac{C}{n^k}$），再令其 $<\varepsilon$
- **分式处理**：处理含 $n$ 的分式时，常用分子有理化、拆项等技巧

### 1.4 收敛数列的性质

- **有界性**：收敛数列必有界，但有界数列不一定收敛
- **保号性**：若 $\lim a_n=A>0$，则从某项起 $a_n>0$
- **四则运算**：极限的和、差、积、商等于极限的和、差、积、商（分母极限不为零）
- **夹逼准则**：$b_n\leq a_n\leq c_n$ 且 $\lim b_n=\lim c_n=A$，则 $\lim a_n=A$
- **单调有界定理**：单调递增有上界的数列必收敛

---

## 2. 函数极限的ε-δ定义

### 2.1 $x\to x_0$ 的极限

$\lim_{x\to x_0}f(x)=A$：$\forall\varepsilon>0$，$\exists\delta>0$，当 $0<|x-x_0|<\delta$ 时，$|f(x)-A|<\varepsilon$。

注意：$0<|x-x_0|$ 表示 $x\neq x_0$，极限与函数在 $x_0$ 处是否有定义无关。

### 2.2 单侧极限

- **右极限**：$\lim_{x\to x_0^+}f(x)=A$，即 $0<x-x_0<\delta$
- **左极限**：$\lim_{x\to x_0^-}f(x)=A$，即 $0<x_0-x<\delta$
- $\lim_{x\to x_0}f(x)=A$ 的充要条件是左极限和右极限都等于 $A$

### 2.3 $x\to\infty$ 的极限

$\lim_{x\to+\infty}f(x)=A$：$\forall\varepsilon>0$，$\exists X>0$，当 $x>X$ 时，$|f(x)-A|<\varepsilon$。

类似地定义 $x\to-\infty$ 和 $x\to\infty$ 的情形。

### 2.4 函数极限的性质

- 唯一性、局部有界性、局部保号性
- 四则运算法则
- 夹逼准则：$g(x)\leq f(x)\leq h(x)$ 且 $\lim g(x)=\lim h(x)=A$，则 $\lim f(x)=A$
- 复合运算法则：若 $\lim_{u\to u_0}f(u)=A$，$\lim_{x\to x_0}\varphi(x)=u_0$ 且 $\varphi(x)\neq u_0$（去心邻域），则 $\lim_{x\to x_0}f[\varphi(x)]=A$

---

## 3. 极限的性质与运算法则

### 3.1 四则运算法则

设 $\lim f(x)=A$，$\lim g(x)=B$，则：
- $\lim[f(x)\pm g(x)]=A\pm B$
- $\lim[f(x)\cdot g(x)]=A\cdot B$
- $\lim\frac{f(x)}{g(x)}=\frac{A}{B}$（$B\neq 0$）
- $\lim[f(x)]^n=[A]^n$，$\lim\sqrt[n]{f(x)}=\sqrt[n]{A}$（$A\geq 0$ 时偶次根号）

### 3.2 有理化技巧

- **分子有理化**：$\frac{\sqrt{x}-\sqrt{a}}{x-a}=\frac{1}{\sqrt{x}+\sqrt{a}}$
- **通分化简**：处理 $\infty-\infty$ 型未定式

### 3.3 夹逼准则的应用

经典例题：$\lim_{n\to\infty}\left(\frac{1}{\sqrt{n^2+1}}+\frac{1}{\sqrt{n^2+2}}+\cdots+\frac{1}{\sqrt{n^2+n}}\right)=1$

利用 $\frac{n}{\sqrt{n^2+n}}\leq S_n\leq\frac{n}{\sqrt{n^2+1}}$，两端极限均为 $1$。

---

## 4. 两个重要极限

### 4.1 第一重要极限

$$\lim_{x\to 0}\frac{\sin x}{x}=1$$

**证明思路**（几何法）：在单位圆中，由面积关系 $\frac{1}{2}\sin x<\frac{1}{2}x<\frac{1}{2}\tan x$，可得 $\cos x<\frac{\sin x}{x}<1$，由夹逼准则得结论。

**推论**：
- $\lim_{x\to 0}\frac{\tan x}{x}=1$
- $\lim_{x\to 0}\frac{1-\cos x}{x^2}=\frac{1}{2}$
- $\lim_{x\to 0}\frac{\arcsin x}{x}=1$，$\lim_{x\to 0}\frac{\arctan x}{x}=1$

### 4.2 第二重要极限

$$\lim_{x\to\infty}\left(1+\frac{1}{x}\right)^x=e\approx 2.71828$$

等价形式：
- $\lim_{x\to 0}(1+x)^{\frac{1}{x}}=e$
- $\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n=e$

**推论**：
- $\lim_{x\to 0}\frac{\ln(1+x)}{x}=1$
- $\lim_{x\to 0}\frac{e^x-1}{x}=1$
- $\lim_{x\to 0}\frac{(1+x)^\alpha-1}{x}=\alpha$

---

## 5. 无穷小与无穷大

### 5.1 无穷小的定义

若 $\lim_{x\to x_0}f(x)=0$，则称 $f(x)$ 为 $x\to x_0$ 时的无穷小量。

**关系**：$\lim f(x)=A$ 当且仅当 $f(x)=A+\alpha(x)$，其中 $\alpha(x)$ 是无穷小。

### 5.2 无穷小阶的比较

设 $\alpha,\beta$ 为同一变化过程中的无穷小：

| 比较 | 定义 | 记号 |
|------|------|------|
| 高阶无穷小 | $\lim\frac{\alpha}{\beta}=0$ | $\alpha=o(\beta)$ |
| 低阶无穷小 | $\lim\frac{\alpha}{\beta}=\infty$ | — |
| 同阶无穷小 | $\lim\frac{\alpha}{\beta}=C\neq 0$ | $\alpha=O(\beta)$ |
| 等价无穷小 | $\lim\frac{\alpha}{\beta}=1$ | $\alpha\sim\beta$ |
| $k$阶无穷小 | $\lim\frac{\alpha}{\beta^k}=C\neq 0$ | — |

### 5.3 常用等价无穷小（$x\to 0$）

- $\sin x\sim x$，$\tan x\sim x$，$\arcsin x\sim x$，$\arctan x\sim x$
- $1-\cos x\sim\frac{1}{2}x^2$，$e^x-1\sim x$，$\ln(1+x)\sim x$
- $(1+x)^\alpha-1\sim\alpha x$
- $a^x-1\sim x\ln a$
- $x-\sin x\sim\frac{1}{6}x^3$

### 5.4 等价无穷小替换定理

若 $\alpha\sim\alpha'$，$\beta\sim\beta'$，且 $\lim\frac{\alpha'}{\beta'}$ 存在，则 $\lim\frac{\alpha}{\beta}=\lim\frac{\alpha'}{\beta'}$。

**注意**：等价替换只适用于乘除因子，不可用于加减因子（加减中的无穷小需另行处理）。

### 5.5 无穷大

若 $\lim_{x\to x_0}|f(x)|=\infty$，则 $f(x)$ 为无穷大量。无穷大与非零无穷小互为倒数关系。

---

## 6. 函数的连续性

### 6.1 连续的定义

函数 $f(x)$ 在 $x_0$ 处连续，需满足以下等价条件之一：
1. $\lim_{x\to x_0}f(x)=f(x_0)$
2. $\lim_{\Delta x\to 0}\Delta y=0$，其中 $\Delta y=f(x_0+\Delta x)-f(x_0)$
3. $\forall\varepsilon>0$，$\exists\delta>0$，当 $|x-x_0|<\delta$ 时，$|f(x)-f(x_0)|<\varepsilon$

### 6.2 左连续与右连续

- **左连续**：$\lim_{x\to x_0^-}f(x)=f(x_0)$
- **右连续**：$\lim_{x\to x_0^+}f(x)=f(x_0)$
- 连续 $\Leftrightarrow$ 左连续且右连续

### 6.3 连续函数的运算

- 连续函数的四则运算结果仍连续（分母不为零时）
- 连续函数的复合函数仍连续
- 初等函数在其定义区间内连续

### 6.4 间断点分类

若 $f(x)$ 在 $x_0$ 处不连续，$x_0$ 为间断点：

**第一类间断点**（左右极限均存在）：
| 类型 | 特征 | 例子 |
|------|------|------|
| 可去间断点 | 左右极限相等但不等于 $f(x_0)$ 或 $f(x_0)$ 无定义 | $\frac{\sin x}{x}$ 在 $x=0$ |
| 跳跃间断点 | 左右极限存在但不相等 | 符号函数 $\text{sgn}(x)$ 在 $x=0$ |

**第二类间断点**（左右极限至少有一个不存在）：
| 类型 | 特征 | 例子 |
|------|------|------|
| 无穷间断点 | 极限为无穷 | $\frac{1}{x}$ 在 $x=0$ |
| 振荡间断点 | 极限不存在（振荡） | $\sin\frac{1}{x}$ 在 $x=0$ |

---

## 7. 闭区间上连续函数的性质

设 $f(x)$ 在 $[a,b]$ 上连续，则有以下重要定理：

### 7.1 有界性定理

$f(x)$ 在 $[a,b]$ 上有界，即 $\exists M>0$，$\forall x\in[a,b]$，$|f(x)|\leq M$。

### 7.2 最值定理

$f(x)$ 在 $[a,b]$ 上必取到最大值和最小值，即 $\exists\xi_1,\xi_2\in[a,b]$，使得 $f(\xi_1)=\max_{[a,b]}f$，$f(\xi_2)=\min_{[a,b]}f$。

### 7.3 零点定理

若 $f(a)\cdot f(b)<0$，则 $\exists\xi\in(a,b)$，使得 $f(\xi)=0$。

应用：证明方程根的存在性。

### 7.4 介值定理

若 $f(a)\neq f(b)$，$\mu$ 为 $f(a)$ 与 $f(b)$ 之间的任意值，则 $\exists\xi\in(a,b)$，使得 $f(\xi)=\mu$。

**推论**：连续函数在闭区间上的值域为 $[\min f, \max f]$。

---

## 8. 本章知识脉络

```
极限
├── 数列极限 ─── ε-N定义 ─── 夹逼准则、单调有界定理
├── 函数极限 ─── ε-δ定义 ─── 单侧极限、x→∞
└── 运算法则 ─── 四则运算、两个重要极限、等价无穷小替换

连续性
├── 定义 ─── lim f(x) = f(x₀)
├── 间断点 ─── 第一类（可去/跳跃）、第二类（无穷/振荡）
└── 闭区间性质 ─── 有界性、最值、零点、介值定理
```
