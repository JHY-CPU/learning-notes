# 02-一元微分学

> 微分学研究函数的变化率与局部线性逼近，是微积分中最具工具性的分支。本章从导数的定义出发，系统介绍求导法则、中值定理、Taylor展开以及函数性态分析。

## 1. 导数的定义与几何意义

### 1.1 导数的定义

$$f'(x_0)=\lim_{\Delta x\to 0}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}=\lim_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}$$

等价条件：
- 右导数 $f'_+(x_0)=\lim_{\Delta x\to 0^+}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$
- 左导数 $f'_-(x_0)=\lim_{\Delta x\to 0^-}\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$
- $f'(x_0)$ 存在 $\Leftrightarrow$ $f'_+(x_0)=f'_-(x_0)$

### 1.2 几何意义

$f'(x_0)$ 是曲线 $y=f(x)$ 在点 $(x_0,f(x_0))$ 处切线的斜率。

- 切线方程：$y-f(x_0)=f'(x_0)(x-x_0)$
- 法线方程：$y-f(x_0)=-\frac{1}{f'(x_0)}(x-x_0)$（$f'(x_0)\neq 0$）

### 1.3 可导与连续的关系

- 可导 $\Rightarrow$ 连续
- 连续 $\nRightarrow$ 可导（如 $f(x)=|x|$ 在 $x=0$ 处连续但不可导）

---

## 2. 求导法则

### 2.1 四则运算法则

设 $u(x),v(x)$ 均可导：
- $(u\pm v)'=u'\pm v'$
- $(uv)'=u'v+uv'$（Leibniz法则）
- $\left(\frac{u}{v}\right)'=\frac{u'v-uv'}{v^2}$（$v\neq 0$）
- $(cu)'=cu'$（常数倍）

### 2.2 链式法则（复合函数求导）

若 $y=f(u)$，$u=\varphi(x)$，则：

$$\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}=f'(u)\cdot\varphi'(x)$$

推广：$y=f(g(h(x)))$，则 $\frac{dy}{dx}=f'(g(h(x)))\cdot g'(h(x))\cdot h'(x)$。

### 2.3 反函数求导

若 $y=f(x)$ 与 $x=\varphi(y)$ 互为反函数，$f'(x)\neq 0$，则：

$$\frac{dy}{dx}=\frac{1}{\frac{dx}{dy}}=\frac{1}{\varphi'(y)}$$

例如：$(\arcsin x)'=\frac{1}{\sqrt{1-x^2}}$，$(\ln x)'=\frac{1}{x}$。

### 2.4 隐函数求导

对 $F(x,y)=0$ 两边关于 $x$ 求导，将 $y$ 视为 $x$ 的函数，利用链式法则。

例如：$x^2+y^2=R^2$，求导得 $2x+2yy'=0$，故 $y'=-\frac{x}{y}$。

### 2.5 参数方程求导

设 $\begin{cases}x=\varphi(t)\\y=\psi(t)\end{cases}$，则：

$$\frac{dy}{dx}=\frac{\psi'(t)}{\varphi'(t)}=\frac{dy/dt}{dx/dt}$$

二阶导数：$\frac{d^2y}{dx^2}=\frac{d}{dx}\left(\frac{dy}{dx}\right)=\frac{\frac{d}{dt}\left(\frac{\psi'(t)}{\varphi'(t)}\right)}{\varphi'(t)}$。

### 2.6 对数求导法

适用于幂指函数 $y=u(x)^{v(x)}$ 或连乘形式。取对数后求导：

$$\ln y=v(x)\ln u(x)\Rightarrow \frac{y'}{y}=v'\ln u+v\cdot\frac{u'}{u}$$

即 $y'=u^v\left(v'\ln u+\frac{vu'}{u}\right)$。

---

## 3. 高阶导数与Leibniz公式

### 3.1 高阶导数

$f''(x)=\frac{d^2y}{dx^2}$，$f^{(n)}(x)=\frac{d^ny}{dx^n}$。

### 3.2 常见函数的n阶导数

| 函数 | n阶导数 |
|------|---------|
| $x^n$ | $n!$（$n$ 阶之后为 $0$） |
| $e^x$ | $e^x$ |
| $a^x$ | $a^x(\ln a)^n$ |
| $\sin x$ | $\sin(x+\frac{n\pi}{2})$ |
| $\cos x$ | $\cos(x+\frac{n\pi}{2})$ |
| $\ln(1+x)$ | $(-1)^{n-1}\frac{(n-1)!}{(1+x)^n}$ |
| $(1+x)^\alpha$ | $\alpha(\alpha-1)\cdots(\alpha-n+1)(1+x)^{\alpha-n}$ |

### 3.3 Leibniz公式

两个函数乘积的高阶导数：

$$(uv)^{(n)}=\sum_{k=0}^{n}\binom{n}{k}u^{(k)}v^{(n-k)}$$

类比二项式展开 $(a+b)^n=\sum\binom{n}{k}a^kb^{n-k}$，记为"类二项式定理"。

---

## 4. 微分的定义与近似计算

### 4.1 微分的定义

若 $\Delta y=f(x_0+\Delta x)-f(x_0)=A\cdot\Delta x+o(\Delta x)$，则称 $dy=A\Delta x$ 为 $f(x)$ 在 $x_0$ 处的微分。

- 可微 $\Leftrightarrow$ 可导，且 $dy=f'(x_0)dx$
- 自变量的微分等于增量：$dx=\Delta x$

### 4.2 微分的几何意义

$dy$ 是曲线 $y=f(x)$ 在点 $(x_0,f(x_0))$ 处切线上纵坐标的增量。

### 4.3 微分在近似计算中的应用

当 $|\Delta x|$ 很小时：

$$f(x_0+\Delta x)\approx f(x_0)+f'(x_0)\Delta x$$

常用近似公式（$|x|$ 很小时）：
- $(1+x)^\alpha\approx 1+\alpha x$
- $e^x\approx 1+x$，$\ln(1+x)\approx x$
- $\sin x\approx x$，$\tan x\approx x$

### 4.4 一阶微分形式不变性

不论 $u$ 是自变量还是中间变量，$dy=f'(u)du$ 的形式不变。

---

## 5. 微分中值定理

### 5.1 Fermat引理

若 $f(x)$ 在 $x_0$ 处可导且取极值，则 $f'(x_0)=0$。

### 5.2 Rolle定理

若 $f(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导，$f(a)=f(b)$，则 $\exists\xi\in(a,b)$，$f'(\xi)=0$。

几何意义：曲线弧的两端等高时，必有水平切线。

### 5.3 Lagrange中值定理

若 $f(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导，则 $\exists\xi\in(a,b)$：

$$f'(\xi)=\frac{f(b)-f(a)}{b-a}$$

等价形式：$f(b)-f(a)=f'(\xi)(b-a)$。

**推论**：若 $f'(x)\equiv 0$（$x\in I$），则 $f(x)=C$（常数）。

### 5.4 Cauchy中值定理

若 $f(x),g(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导，$g'(x)\neq 0$，则 $\exists\xi\in(a,b)$：

$$\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(\xi)}{g'(\xi)}$$

Rolle $\subset$ Lagrange $\subset$ Cauchy（Lagrange 是 $g(x)=x$ 的特例，Rolle 是 $f(a)=f(b)$ 的特例）。

---

## 6. Taylor公式与Taylor展开

### 6.1 Taylor公式（带Peano余项）

若 $f(x)$ 在 $x_0$ 处有 $n$ 阶导数，则：

$$f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\cdots+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+o((x-x_0)^n)$$

### 6.2 Taylor公式（带Lagrange余项）

$$f(x)=\sum_{k=0}^{n}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k+\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$$

其中 $\xi$ 在 $x_0$ 与 $x$ 之间。

### 6.3 Maclaurin展开式（$x_0=0$ 的特殊情形）

### 6.4 常见函数的Taylor展开式

$$e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots+\frac{x^n}{n!}+o(x^n)$$

$$\sin x=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots+(-1)^n\frac{x^{2n+1}}{(2n+1)!}+o(x^{2n+1})$$

$$\cos x=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\cdots+(-1)^n\frac{x^{2n}}{(2n)!}+o(x^{2n})$$

$$\ln(1+x)=x-\frac{x^2}{2}+\frac{x^3}{3}-\cdots+(-1)^{n-1}\frac{x^n}{n}+o(x^n)\quad(|x|<1)$$

$$(1+x)^\alpha=1+\alpha x+\frac{\alpha(\alpha-1)}{2!}x^2+\cdots+\frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}x^n+o(x^n)$$

特别地：$\frac{1}{1-x}=1+x+x^2+\cdots+x^n+o(x^n)$，$\frac{1}{1+x}=1-x+x^2-\cdots+(-1)^nx^n+o(x^n)$。

### 6.5 Taylor展开的常用技巧

- **间接展开**：利用已知展开式和四则运算、复合运算得到新展开式
- **代入法**：将 $u=-x^2$ 代入 $\frac{1}{1-u}$ 的展开式
- **求导/积分法**：对 $\frac{1}{1+x}$ 积分得 $\ln(1+x)$，对 $\sin x$ 求导得 $\cos x$

---

## 7. L'Hôpital法则

### 7.1 基本形式

$\frac{0}{0}$ 型：若 $f(x_0)=g(x_0)=0$，且 $\lim_{x\to x_0}\frac{f'(x)}{g'(x)}$ 存在（或为 $\infty$），则：

$$\lim_{x\to x_0}\frac{f(x)}{g(x)}=\lim_{x\to x_0}\frac{f'(x)}{g'(x)}$$

$\frac{\infty}{\infty}$ 型同理，对 $x\to x_0$ 和 $x\to\infty$ 均适用。

### 7.2 其他未定式的处理

| 类型 | 转化方法 |
|------|----------|
| $0\cdot\infty$ | 化为 $\frac{0}{1/\infty}$ 或 $\frac{\infty}{1/0}$ |
| $\infty-\infty$ | 通分或有理化 |
| $0^0,1^\infty,\infty^0$ | 取对数化为 $0\cdot\ln 0$ 等形式 |

### 7.3 使用注意事项

- 必须验证是未定式才能使用
- 若求导后仍为未定式，可继续使用
- 有时需配合等价无穷小替换简化计算
- L'Hôpital 法则的逆命题不成立

---

## 8. 函数单调性与极值判别

### 8.1 单调性判别

设 $f(x)$ 在 $[a,b]$ 上连续，在 $(a,b)$ 内可导：
- $f'(x)>0 \Rightarrow f(x)$ 严格递增
- $f'(x)<0 \Rightarrow f(x)$ 严格递减
- $f'(x)\geq 0 \Leftrightarrow f(x)$ 单调递增（非严格）

**驻点**：$f'(x_0)=0$ 的点，可能为极值点或拐点。

### 8.2 极值的第一充分条件

若 $f'(x)$ 在 $x_0$ 的两侧变号：
- $f'$ 由正变负 $\Rightarrow$ $x_0$ 为极大值点
- $f'$ 由负变正 $\Rightarrow$ $x_0$ 为极小值点

### 8.3 极值的第二充分条件

若 $f'(x_0)=0$ 且 $f''(x_0)\neq 0$：
- $f''(x_0)<0 \Rightarrow$ 极大值
- $f''(x_0)>0 \Rightarrow$ 极小值

### 8.4 最值问题

- 闭区间 $[a,b]$ 上连续函数的最值：比较驻点、不可导点、端点的函数值
- 开区间上的最值：结合极限行为判断

---

## 9. 凹凸性与拐点

### 9.1 凹凸性定义

- **上凸（凹向上）**：$f''(x)>0$，曲线弧位于切线上方
- **下凸（凹向下）**：$f''(x)<0$，曲线弧位于切线下方

### 9.2 拐点

曲线凹凸性改变的点。$f''(x_0)=0$ 或 $f''(x_0)$ 不存在是拐点的必要条件（非充分条件）。需检查 $f''(x)$ 在 $x_0$ 两侧是否变号。

### 9.3 渐近线

- **水平渐近线**：$y=A$，若 $\lim_{x\to\pm\infty}f(x)=A$
- **垂直渐近线**：$x=x_0$，若 $\lim_{x\to x_0}f(x)=\infty$
- **斜渐近线**：$y=kx+b$，其中 $k=\lim\frac{f(x)}{x}$，$b=\lim[f(x)-kx]$

---

## 10. 曲率与曲率半径

### 10.1 曲率公式

$$K=\frac{|y''|}{(1+y'^2)^{3/2}}$$

参数方程：$K=\frac{|x'y''-x''y'|}{(x'^2+y'^2)^{3/2}}$。

### 10.2 曲率半径

$$R=\frac{1}{K}=\frac{(1+y'^2)^{3/2}}{|y''|}$$

曲率半径越大，曲线弯曲程度越小。

---

## 11. 知识脉络

```
导数
├── 定义 ─── 极限形式 ─── 可导与连续的关系
├── 求导法则 ─── 四则运算、链式法则、反函数、隐函数、参数方程、对数求导
├── 高阶导数 ─── Leibniz公式
└── 微分 ─── 近似计算、一阶微分形式不变性

中值定理
├── Rolle → Lagrange → Cauchy（层层推广）
└── Taylor公式 ─── 带余项展开、常用展开式

函数分析
├── L'Hôpital法则 ─── 各类未定式
├── 单调性与极值 ─── 一阶/二阶判别
├── 凹凸性与拐点 ─── 二阶导数判别
└── 曲率 ───  曲率公式与曲率半径
```
