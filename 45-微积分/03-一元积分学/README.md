# 03-一元积分学

> 积分是微分的逆运算，也是从局部到整体的桥梁。本章系统介绍不定积分的计算技巧、定积分的理论与应用，以及反常积分和特殊函数。

## 1. 不定积分的定义与基本积分表

### 1.1 原函数与不定积分

若在区间 $I$ 上 $F'(x)=f(x)$，则称 $F(x)$ 为 $f(x)$ 的一个原函数。

$$\int f(x)\,dx=F(x)+C$$

其中 $C$ 为任意常数。不定积分表示一族函数，其导数均为 $f(x)$。

### 1.2 不定积分的基本性质

- $\int[f(x)\pm g(x)]\,dx=\int f(x)\,dx\pm\int g(x)\,dx$
- $\int kf(x)\,dx=k\int f(x)\,dx$（$k$ 为非零常数）
- $\left[\int f(x)\,dx\right]'=f(x)$；$\int f'(x)\,dx=f(x)+C$

### 1.3 基本积分表

| 函数 $f(x)$ | 不定积分 $\int f(x)\,dx$ |
|---|---|
| $x^n$（$n\neq -1$） | $\frac{x^{n+1}}{n+1}+C$ |
| $\frac{1}{x}$ | $\ln\|x\|+C$ |
| $e^x$ | $e^x+C$ |
| $a^x$ | $\frac{a^x}{\ln a}+C$ |
| $\sin x$ | $-\cos x+C$ |
| $\cos x$ | $\sin x+C$ |
| $\sec^2 x$ | $\tan x+C$ |
| $\csc^2 x$ | $-\cot x+C$ |
| $\frac{1}{1+x^2}$ | $\arctan x+C$ |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x+C$ |
| $\sec x\tan x$ | $\sec x+C$ |
| $\csc x\cot x$ | $-\csc x+C$ |

---

## 2. 换元积分法

### 2.1 第一类换元法（凑微分法）

$$\int f[\varphi(x)]\varphi'(x)\,dx=\int f(u)\,du\Big|_{u=\varphi(x)}$$

**核心思想**：将被积表达式凑成 $f[\varphi(x)]d[\varphi(x)]$ 的形式。

**常用凑微分公式**：
- $\int f(ax+b)\,dx=\frac{1}{a}\int f(ax+b)\,d(ax+b)$
- $\int f(x^n)x^{n-1}\,dx=\frac{1}{n}\int f(u)\,du\quad(u=x^n)$
- $\int f(\sin x)\cos x\,dx=\int f(\sin x)\,d(\sin x)$
- $\int f(\ln x)\frac{1}{x}\,dx=\int f(\ln x)\,d(\ln x)$
- $\int f(e^x)e^x\,dx=\int f(e^x)\,d(e^x)$
- $\int f(\tan x)\sec^2 x\,dx=\int f(\tan x)\,d(\tan x)$

### 2.2 第二类换元法（变量替换法）

令 $x=\varphi(t)$，其中 $\varphi(t)$ 单调可导且 $\varphi'(t)\neq 0$，则：

$$\int f(x)\,dx=\int f[\varphi(t)]\varphi'(t)\,dt\Big|_{t=\varphi^{-1}(x)}$$

**常用替换**：
| 含根号形式 | 替换 | 替换后三角关系 |
|---|---|---|
| $\sqrt{a^2-x^2}$ | $x=a\sin t$ | $\sqrt{a^2-x^2}=a\cos t$ |
| $\sqrt{a^2+x^2}$ | $x=a\tan t$ | $\sqrt{a^2+x^2}=a\sec t$ |
| $\sqrt{x^2-a^2}$ | $x=a\sec t$ | $\sqrt{x^2-a^2}=a\tan t$ |
| $\sqrt{ax+b}$ | $t=\sqrt{ax+b}$ | $x=\frac{t^2-b}{a}$ |
| $\sqrt[n]{\frac{ax+b}{cx+d}}$ | $t=\sqrt[n]{\frac{ax+b}{cx+d}}$ | — |

---

## 3. 分部积分法

### 3.1 公式

$$\int u\,dv=uv-\int v\,du$$

### 3.2 选 $u$ 的LIATE原则

按优先级选择 $u$，剩下的为 $dv$：
1. **L**ogarithmic（对数函数）：$\ln x$, $\log_a x$
2. **I**nverse trig（反三角函数）：$\arcsin x$, $\arctan x$
3. **A**lgebraic（代数函数）：$x^n$, 多项式
4. **T**rigonometric（三角函数）：$\sin x$, $\cos x$
5. **E**xponential（指数函数）：$e^x$, $a^x$

### 3.3 典型应用

- $\int x^n e^x\,dx$：反复分部，$n$ 次后消去 $x^n$
- $\int x^n\sin x\,dx$：同上
- $\int e^x\sin x\,dx$：分部两次后出现循环，解方程
- $\int \ln x\,dx$：$u=\ln x$，$dv=dx$
- $\int x\arctan x\,dx$：$u=\arctan x$，$dv=x\,dx$

### 3.4 递推公式

分部积分可建立递推关系。例如：

$$I_n=\int \sin^n x\,dx=-\frac{1}{n}\sin^{n-1}x\cos x+\frac{n-1}{n}I_{n-2}$$

---

## 4. 有理函数积分

### 4.1 部分分式分解

有理函数 $\frac{P(x)}{Q(x)}$（$\deg P<\deg Q$）可分解为部分分式之和：

分解规则（按 $Q(x)$ 的因式类型）：
| $Q(x)$ 的因式 | 部分分式项 |
|---|---|
| $(x-a)$ | $\frac{A}{x-a}$ |
| $(x-a)^k$ | $\frac{A_1}{x-a}+\frac{A_2}{(x-a)^2}+\cdots+\frac{A_k}{(x-a)^k}$ |
| $(x^2+px+q)$（不可约） | $\frac{Ax+B}{x^2+px+q}$ |
| $(x^2+px+q)^k$ | $\frac{A_1x+B_1}{x^2+px+q}+\cdots+\frac{A_kx+B_k}{(x^2+px+q)^k}$ |

### 4.2 三角有理函数

用万能替换 $t=\tan\frac{x}{2}$：
$$\sin x=\frac{2t}{1+t^2},\quad \cos x=\frac{1-t^2}{1+t^2},\quad dx=\frac{2\,dt}{1+t^2}$$

将三角有理函数化为关于 $t$ 的有理函数。

### 4.3 某些无理函数的积分

- 含 $\sqrt[n]{ax+b}$：令 $t=\sqrt[n]{ax+b}$
- 含 $\sqrt[n]{\frac{ax+b}{cx+d}}$：令 $t=\sqrt[n]{\frac{ax+b}{cx+d}}$
- Euler 替换：用于含 $\sqrt{ax^2+bx+c}$ 的情形

---

## 5. 定积分的定义（Riemann积分）

### 5.1 分割

将 $[a,b]$ 分成 $n$ 个小区间：$a=x_0<x_1<\cdots<x_n=b$，每个小区间长度 $\Delta x_i=x_i-x_{i-1}$。

### 5.2 Riemann和

在每个小区间 $[x_{i-1},x_i]$ 中任取一点 $\xi_i$，作和式：

$$S=\sum_{i=1}^{n}f(\xi_i)\Delta x_i$$

### 5.3 Riemann积分的定义

$$\int_a^bf(x)\,dx=\lim_{\lambda\to 0}\sum_{i=1}^{n}f(\xi_i)\Delta x_i$$

其中 $\lambda=\max\{\Delta x_i\}$ 为最大子区间长度。

**可积条件**：$f(x)$ 在 $[a,b]$ 上连续，或仅有有限个第一类间断点，则 $f(x)$ 在 $[a,b]$ 上可积。

---

## 6. 定积分的性质

### 6.1 基本性质

- **线性**：$\int_a^b[\alpha f(x)+\beta g(x)]\,dx=\alpha\int_a^bf(x)\,dx+\beta\int_a^bg(x)\,dx$
- **区间可加**：$\int_a^bf\,dx=\int_a^cf\,dx+\int_a^bf\,dx$
- **保号性**：若 $f(x)\geq g(x)$，则 $\int_a^bf\,dx\geq\int_a^bg\,dx$
- **估值不等式**：$m(b-a)\leq\int_a^bf(x)\,dx\leq M(b-a)$，其中 $m,M$ 为 $f$ 在 $[a,b]$ 上的最小最大值

### 6.2 积分中值定理

若 $f(x)$ 在 $[a,b]$ 上连续，则 $\exists\xi\in[a,b]$：

$$\int_a^bf(x)\,dx=f(\xi)(b-a)$$

$f(\xi)=\frac{1}{b-a}\int_a^bf(x)\,dx$ 称为 $f$ 在 $[a,b]$ 上的平均值。

### 6.3 对称区间

- 若 $f(x)$ 为奇函数：$\int_{-a}^{a}f(x)\,dx=0$
- 若 $f(x)$ 为偶函数：$\int_{-a}^{a}f(x)\,dx=2\int_0^{a}f(x)\,dx$

---

## 7. 微积分基本定理（Newton-Leibniz公式）

### 7.1 变上限积分函数

$$\Phi(x)=\int_a^xf(t)\,dt$$

若 $f(t)$ 在 $[a,b]$ 上连续，则 $\Phi'(x)=f(x)$，即 $\Phi(x)$ 是 $f(x)$ 的一个原函数。

### 7.2 Newton-Leibniz公式

$$\int_a^bf(x)\,dx=F(b)-F(a)\triangleq F(x)\Big|_a^b$$

其中 $F(x)$ 是 $f(x)$ 的任意一个原函数。

**意义**：将定积分的计算转化为求原函数，建立了微分与积分之间的桥梁。

### 7.3 推广

$$\frac{d}{dx}\int_{a(x)}^{b(x)}f(t)\,dt=f[b(x)]\cdot b'(x)-f[a(x)]\cdot a'(x)$$

---

## 8. 定积分的计算

### 8.1 换元法

令 $x=\varphi(t)$，$\varphi(\alpha)=a$，$\varphi(\beta)=b$，则：

$$\int_a^bf(x)\,dx=\int_\alpha^\beta f[\varphi(t)]\varphi'(t)\,dt$$

**注意**：换元时必须同时变换积分上下限。

### 8.2 分部积分法

$$\int_a^bu\,dv=uv\Big|_a^b-\int_a^bv\,du$$

### 8.3 常用积分公式

- $\int_0^{\pi/2}\sin^n x\,dx=\int_0^{\pi/2}\cos^n x\,dx=\begin{cases}\frac{(n-1)!!}{n!!}\cdot\frac{\pi}{2} & n\text{ 偶}\\\frac{(n-1)!!}{n!!} & n\text{ 奇}\end{cases}$
- $\int_0^{\pi}\sin^n x\,dx=2\int_0^{\pi/2}\sin^n x\,dx$（$n$ 为正整数）
- Wallis 公式：$\frac{\pi}{2}=\prod_{n=1}^{\infty}\frac{(2n)^2}{(2n-1)(2n+1)}$

---

## 9. 反常积分

### 9.1 无穷区间上的反常积分

$$\int_a^{+\infty}f(x)\,dx=\lim_{t\to+\infty}\int_a^tf(x)\,dx$$

收敛条件：若极限存在则收敛，否则发散。类似定义 $\int_{-\infty}^b$ 和 $\int_{-\infty}^{+\infty}$。

### 9.2 无界函数的反常积分（瑕积分）

若 $f(x)$ 在 $x=a$ 的右邻域无界（$a$ 为瑕点）：

$$\int_a^bf(x)\,dx=\lim_{\varepsilon\to 0^+}\int_{a+\varepsilon}^bf(x)\,dx$$

### 9.3 收敛判别法（比较判别法）

- 若 $0\leq f(x)\leq g(x)$ 且 $\int g\,dx$ 收敛，则 $\int f\,dx$ 收敛
- **极限形式**：$\lim_{x\to+\infty}\frac{f(x)}{g(x)}=C$（$0<C<\infty$），则两者同敛散

### 9.4 常见反常积分

- $\int_1^{+\infty}\frac{1}{x^p}\,dx$：$p>1$ 收敛，$p\leq 1$ 发散
- $\int_0^1\frac{1}{x^p}\,dx$：$p<1$ 收敛，$p\geq 1$ 发散
- $\int_0^{+\infty}e^{-x^2}\,dx=\frac{\sqrt{\pi}}{2}$（Gauss积分）

---

## 10. 定积分的应用

### 10.1 平面图形面积

- 直角坐标：$S=\int_a^b|f(x)-g(x)|\,dx$
- 参数方程：$S=\int_\alpha^\beta|y(t)x'(t)|\,dt$
- 极坐标：$S=\frac{1}{2}\int_\alpha^\beta r^2(\theta)\,d\theta$

### 10.2 旋转体体积

- 绕 $x$ 轴：$V=\pi\int_a^b[f(x)]^2\,dx$
- 绕 $y$ 轴：$V=2\pi\int_a^bx|f(x)|\,dx$（柱壳法）
- 平行截面面积已知：$V=\int_a^bA(x)\,dx$

### 10.3 弧长

$$L=\int_a^b\sqrt{1+[f'(x)]^2}\,dx$$

参数方程：$L=\int_\alpha^\beta\sqrt{[x'(t)]^2+[y'(t)]^2}\,dt$

极坐标：$L=\int_\alpha^\beta\sqrt{r^2+r'^2}\,d\theta$

### 10.4 旋转曲面面积

$$S=2\pi\int_a^b|f(x)|\sqrt{1+[f'(x)]^2}\,dx$$

---

## 11. Gamma函数与Beta函数

### 11.1 Gamma函数

$$\Gamma(s)=\int_0^{+\infty}x^{s-1}e^{-x}\,dx\quad(s>0)$$

**基本性质**：
- $\Gamma(1)=1$
- $\Gamma(s+1)=s\Gamma(s)$（递推公式）
- $\Gamma(n+1)=n!$（$n$ 为正整数）
- $\Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}$

### 11.2 Beta函数

$$B(p,q)=\int_0^1x^{p-1}(1-x)^{q-1}\,dx\quad(p>0,q>0)$$

**基本性质**：
- $B(p,q)=B(q,p)$（对称性）
- $B(p,q)=\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}$（与 Gamma 的关系）
- $B(p,q)=2\int_0^{\pi/2}\sin^{2p-1}\theta\cos^{2q-1}\theta\,d\theta$（三角形式）

### 11.3 应用

- 计算反常积分：$\int_0^{+\infty}\frac{x^{s-1}}{1+x}\,dx=\frac{\pi}{\sin(\pi s)}$
- 概率分布：Gamma分布与Beta分布的归一化常数
- 统计学：$t$ 分布、$F$ 分布的密度函数涉及 Beta 函数

---

## 12. 知识脉络

```
不定积分
├── 定义 ─── 原函数、基本积分表
├── 换元法 ─── 第一类（凑微分）、第二类（变量替换）
├── 分部积分 ─── LIATE原则、递推公式
└── 特殊类型 ─── 有理函数、三角有理函数

定积分
├── 定义 ─── Riemann和
├── 性质 ─── 线性、区间可加、中值定理
├── Newton-Leibniz ─── 微积分基本定理
├── 计算 ─── 换元、分部
└── 应用 ─── 面积、体积、弧长

反常积分
├── 无穷区间 ─── 比较判别法
├── 瑕积分 ─── 瑕点处理
└── 特殊函数 ─── Gamma、Beta函数
```
