# 05-级数与变换

> 级数将有限求和推广到无穷，变换将函数从时域映射到频域。本章涵盖数项级数与函数项级数的收敛理论、Fourier级数与变换、Laplace变换、Z变换，以及在信号处理和深度学习中的应用。

## 1. 数项级数（收敛判别法）

### 1.1 级数的基本概念

无穷级数 $\sum_{n=1}^{\infty}a_n=a_1+a_2+\cdots$，部分和 $S_n=\sum_{k=1}^{n}a_k$。

级数收敛 $\Leftrightarrow$ $\lim_{n\to\infty}S_n$ 存在；级数发散 $\Leftrightarrow$ $\lim_{n\to\infty}S_n$ 不存在。

### 1.2 收敛的必要条件

若 $\sum a_n$ 收敛，则 $\lim_{n\to\infty}a_n=0$。

**注意**：$\lim a_n=0$ 是必要条件而非充分条件（调和级数 $\sum\frac{1}{n}$ 发散）。

### 1.3 收敛级数的性质

- 若 $\sum a_n$、$\sum b_n$ 收敛，则 $\sum(\alpha a_n+\beta b_n)$ 收敛
- 改变有限项不改变级数的敛散性
- 收敛级数加括号后仍收敛（反之不成立，除非各项同号）

### 1.4 常用判别法

**比较判别法**（正项级数）：
- 若 $0\leq a_n\leq b_n$，$\sum b_n$ 收敛则 $\sum a_n$ 收敛
- 极限形式：$\lim_{n\to\infty}\frac{a_n}{b_n}=C$（$0<C<\infty$），则同敛散

**比值判别法（d'Alembert）**：

$$\rho=\lim_{n\to\infty}\frac{a_{n+1}}{a_n}$$

- $\rho<1$：收敛
- $\rho>1$：发散
- $\rho=1$：不确定

**根值判别法（Cauchy）**：

$$\rho=\lim_{n\to\infty}\sqrt[n]{a_n}$$

判别准则同比值判别法。

**积分判别法（Cauchy）**：

若 $f(x)$ 在 $[1,+\infty)$ 上非负、单调递减，且 $f(n)=a_n$，则 $\sum a_n$ 与 $\int_1^{+\infty}f(x)\,dx$ 同敛散。

---

## 2. 正项级数与交错级数

### 2.1 正项级数

正项级数部分和 $S_n$ 单调递增，收敛 $\Leftrightarrow$ $S_n$ 有上界。

**常用判别法总结**：

| 级数形式 | 适用判别法 |
|---|---|
| $a_n$ 含 $n!$ | 比值判别法 |
| $a_n$ 含 $n$ 次幂 | 根值判别法 |
| $a_n$ 含积分形式 | 积分判别法 |
| $a_n$ 与已知级数可比 | 比较判别法 |

**常用基准级数**：
- $p$-级数：$\sum\frac{1}{n^p}$，$p>1$ 收敛，$p\leq 1$ 发散
- 几何级数：$\sum r^n$，$|r|<1$ 收敛（和为 $\frac{1}{1-r}$）

### 2.2 交错级数

**Leibniz判别法**：对于交错级数 $\sum(-1)^{n-1}a_n$（$a_n>0$），若：
1. $a_n$ 单调递减：$a_{n+1}\leq a_n$
2. $\lim_{n\to\infty}a_n=0$

则级数收敛，且 $|S-S_n|\leq a_{n+1}$（余项估计）。

**典型例子**：
- 交错调和级数：$\sum_{n=1}^{\infty}\frac{(-1)^{n-1}}{n}=\ln 2$
- 交错 $p$-级数：$\sum_{n=1}^{\infty}\frac{(-1)^{n-1}}{n^p}$（$p>0$ 均收敛）

---

## 3. 绝对收敛与条件收敛

### 3.1 定义

- **绝对收敛**：$\sum|a_n|$ 收敛 $\Rightarrow$ $\sum a_n$ 收敛
- **条件收敛**：$\sum a_n$ 收敛但 $\sum|a_n|$ 发散

### 3.2 绝对收敛的性质

- 绝对收敛级数可以任意重排，和不变
- 条件收敛级数经适当重排可以收敛到任意值（Riemann 重排定理）

### 3.3 判别流程

```
判断级数 ∑aₙ 的敛散性
├── 检查 lim aₙ = 0？若不成立则发散
├── 判断 ∑|aₙ| 的敛散性
│   ├── 收敛 → 绝对收敛
│   └── 发散 → 转向判断 ∑aₙ（可能条件收敛）
│       ├── 正项级数 → 比较/比值/根值/积分判别法
│       └── 交错级数 → Leibniz判别法
└── 记录结论：发散 / 条件收敛 / 绝对收敛
```

---

## 4. 幂级数与收敛半径

### 4.1 幂级数的定义

$$\sum_{n=0}^{\infty}a_n(x-x_0)^n=a_0+a_1(x-x_0)+a_2(x-x_0)^2+\cdots$$

### 4.2 收敛半径（Cauchy-Hadamard公式）

$$R=\frac{1}{\limsup_{n\to\infty}\sqrt[n]{|a_n|}}$$

等价形式：$R=\lim_{n\to\infty}\left|\frac{a_n}{a_{n+1}}\right|$（当此极限存在时）。

**收敛区间**：$(x_0-R, x_0+R)$，端点需单独判断。

| 情形 | 收敛域 |
|---|---|
| $R=0$ | 仅在 $x=x_0$ 收敛 |
| $0<R<\infty$ | 需检查两个端点 |
| $R=\infty$ | 在整个实数轴收敛 |

### 4.3 幂级数的运算性质

**逐项求导**：

$$\left[\sum_{n=0}^{\infty}a_n(x-x_0)^n\right]'=\sum_{n=1}^{\infty}na_n(x-x_0)^{n-1}$$

收敛半径不变（端点可能改变）。

**逐项积分**：

$$\int_{x_0}^{x}\left[\sum_{n=0}^{\infty}a_n(t-x_0)^n\right]dt=\sum_{n=0}^{\infty}\frac{a_n}{n+1}(x-x_0)^{n+1}$$

收敛半径不变。

**幂级数的乘法**：

$$\left(\sum a_nx^n\right)\left(\sum b_nx^n\right)=\sum c_nx^n, \quad c_n=\sum_{k=0}^{n}a_kb_{n-k}$$

---

## 5. 函数展开为幂级数

### 5.1 Taylor级数

若 $f(x)$ 在 $x_0$ 处有任意阶导数，则：

$$f(x)\sim\sum_{n=0}^{\infty}\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n$$

$f(x)$ 等于其 Taylor 级数（在收敛域内）的充要条件：余项趋于零。

### 5.2 常见函数的幂级数展开

$$e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!},\quad x\in(-\infty,+\infty)$$

$$\sin x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n+1}}{(2n+1)!},\quad x\in(-\infty,+\infty)$$

$$\cos x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n}}{(2n)!},\quad x\in(-\infty,+\infty)$$

$$\ln(1+x)=\sum_{n=1}^{\infty}\frac{(-1)^{n-1}x^n}{n},\quad x\in(-1,1]$$

$$\frac{1}{1-x}=\sum_{n=0}^{\infty}x^n,\quad x\in(-1,1)$$

$$(1+x)^\alpha=\sum_{n=0}^{\infty}\binom{\alpha}{n}x^n,\quad x\in(-1,1)$$

$$\arctan x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n+1}}{2n+1},\quad x\in[-1,1]$$

### 5.3 间接展开方法

- **代入法**：将 $u=-x^2$ 代入 $\frac{1}{1-u}$ 得 $\frac{1}{1+x^2}=\sum(-1)^nx^{2n}$
- **逐项求导**：对 $\frac{1}{1+x}$ 求导得 $\frac{-1}{(1+x)^2}$
- **逐项积分**：对 $\frac{1}{1+x}$ 积分得 $\ln(1+x)$
- **四则运算**：利用已知展开式通过加减乘除构造

---

## 6. Fourier级数

### 6.1 三角级数

$$f(x)\sim\frac{a_0}{2}+\sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)$$

### 6.2 Fourier系数

$$a_0=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\,dx$$

$$a_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos nx\,dx\quad(n=1,2,\ldots)$$

$$b_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin nx\,dx\quad(n=1,2,\ldots)$$

对于周期为 $2l$ 的函数，将积分区间和自变量适当缩放即可。

### 6.3 Dirichlet收敛条件

若 $f(x)$ 在 $[-\pi,\pi]$ 上：
- 连续或仅有有限个第一类间断点
- 至多有有限个极值点

则 Fourier 级数收敛：
- 在连续点处，收敛于 $f(x)$
- 在间断点 $x_0$ 处，收敛于 $\frac{f(x_0^-)+f(x_0^+)}{2}$

### 6.4 奇偶函数的Fourier级数

- **偶函数**：$b_n=0$，只有余弦项（Fourier 余弦级数）
- **奇函数**：$a_n=0$，只有正弦项（Fourier 正弦级数）

### 6.5 Parseval等式

$$\frac{1}{\pi}\int_{-\pi}^{\pi}[f(x)]^2\,dx=\frac{a_0^2}{2}+\sum_{n=1}^{\infty}(a_n^2+b_n^2)$$

物理意义：信号的能量等于各频率分量能量之和。

---

## 7. Fourier变换与逆变换

### 7.1 连续Fourier变换

$$F(\omega)=\hat{f}(\omega)=\int_{-\infty}^{+\infty}f(t)e^{-i\omega t}\,dt$$

$$f(t)=\check{F}(t)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}F(\omega)e^{i\omega t}\,d\omega$$

### 7.2 基本性质

| 性质 | 时域 | 频域 |
|---|---|---|
| 线性 | $\alpha f+\beta g$ | $\alpha\hat{f}+\beta\hat{g}$ |
| 时移 | $f(t-t_0)$ | $e^{-i\omega t_0}\hat{f}(\omega)$ |
| 频移 | $e^{i\omega_0 t}f(t)$ | $\hat{f}(\omega-\omega_0)$ |
| 尺度变换 | $f(at)$ | $\frac{1}{|a|}\hat{f}\left(\frac{\omega}{a}\right)$ |
| 微分 | $f'(t)$ | $i\omega\hat{f}(\omega)$ |
| 卷积 | $f*g$ | $\hat{f}\cdot\hat{g}$ |
| 乘积 | $f\cdot g$ | $\frac{1}{2\pi}\hat{f}*\hat{g}$ |

### 7.3 卷积定理

$$(f*g)(t)=\int_{-\infty}^{+\infty}f(\tau)g(t-\tau)\,d\tau$$

时域卷积等于频域乘积：$\widehat{f*g}=\hat{f}\cdot\hat{g}$。

### 7.4 应用

- 频谱分析：将信号分解为不同频率的正弦波
- 滤波：在频域设计滤波器
- 信号压缩：保留主要频率分量

---

## 8. Laplace变换与逆变换

### 8.1 定义

$$F(s)=\mathcal{L}\{f(t)\}=\int_0^{+\infty}f(t)e^{-st}\,dt$$

其中 $s=\sigma+i\omega$ 为复频率变量。

### 8.2 收敛域

$\text{Re}(s)>\sigma_0$（收敛横坐标）时积分绝对收敛。

### 8.3 常见Laplace变换对

| $f(t)$ | $F(s)$ |
|---|---|
| $1$ | $\frac{1}{s}$ |
| $t^n$ | $\frac{n!}{s^{n+1}}$ |
| $e^{at}$ | $\frac{1}{s-a}$ |
| $\sin\omega t$ | $\frac{\omega}{s^2+\omega^2}$ |
| $\cos\omega t$ | $\frac{s}{s^2+\omega^2}$ |
| $e^{at}\sin\omega t$ | $\frac{\omega}{(s-a)^2+\omega^2}$ |
| $t^n e^{at}$ | $\frac{n!}{(s-a)^{n+1}}$ |

### 8.4 Laplace变换的性质

- **线性**：$\mathcal{L}\{\alpha f+\beta g\}=\alpha F(s)+\beta G(s)$
- **时移**：$\mathcal{L}\{f(t-a)u(t-a)\}=e^{-as}F(s)$
- **s域微分**：$\mathcal{L}\{tf(t)\}=-F'(s)$
- **时域微分**：$\mathcal{L}\{f'(t)\}=sF(s)-f(0)$
- **时域积分**：$\mathcal{L}\left\{\int_0^tf(\tau)d\tau\right\}=\frac{F(s)}{s}$
- **卷积**：$\mathcal{L}\{f*g\}=F(s)G(s)$

### 8.5 逆Laplace变换

$$f(t)=\mathcal{L}^{-1}\{F(s)\}=\frac{1}{2\pi i}\int_{\sigma-i\infty}^{\sigma+i\infty}F(s)e^{st}\,ds$$

实际计算常用部分分式分解法：将 $F(s)$ 分解为基本形式的和，查表求逆变换。

### 8.6 应用：解微分方程

对常微分方程做 Laplace 变换，将微分方程转化为代数方程，求解后再做逆变换。

---

## 9. Z变换基础

### 9.1 定义

离散信号 $x[n]$ 的 Z 变换：

$$X(z)=\sum_{n=-\infty}^{+\infty}x[n]z^{-n}$$

其中 $z$ 为复变量。

### 9.2 与Laplace变换的关系

Z 变换是 Laplace 变换在离散时间域的对应。若 $s=\sigma+i\omega$，令 $z=e^{sT}$（$T$ 为采样周期），则：

$$X(z)\Big|_{z=e^{sT}}=X_{\text{Laplace}}(s)$$

### 9.3 常见Z变换对

| $x[n]$ | $X(z)$ | 收敛域 |
|---|---|---|
| $\delta[n]$ | $1$ | 全平面 |
| $u[n]$（单位阶跃） | $\frac{z}{z-1}$ | $|z|>1$ |
| $a^n u[n]$ | $\frac{z}{z-a}$ | $|z|>|a|$ |
| $na^n u[n]$ | $\frac{az}{(z-a)^2}$ | $|z|>|a|$ |
| $e^{j\omega_0 n}u[n]$ | $\frac{z}{z-e^{j\omega_0}}$ | $|z|>1$ |

### 9.4 Z变换的性质

- **线性**：$Z\{\alpha x_1+\beta x_2\}=\alpha X_1(z)+\beta X_2(z)$
- **时移**：$Z\{x[n-k]\}=z^{-k}X(z)$
- **卷积**：$Z\{x_1*x_2\}=X_1(z)X_2(z)$
- **初值定理**：$x[0]=\lim_{z\to\infty}X(z)$
- **终值定理**：$\lim_{n\to\infty}x[n]=\lim_{z\to 1}(1-z^{-1})X(z)$

### 9.5 应用

- 数字滤波器的设计与分析
- 离散系统的传递函数
- 离散控制系统的稳定性分析（极点在单位圆内则稳定）

---

## 10. 在信号处理与深度学习中的应用

### 10.1 频谱分析

Fourier 变换将信号从时域映射到频域：
- **音频处理**：识别不同频率成分，实现降噪、均衡
- **图像处理**：二维 Fourier 变换用于频率域滤波
- **频谱图**：短时 Fourier 变换（STFT）分析非平稳信号

### 10.2 卷积定理的应用

- **卷积神经网络（CNN）**：空间域的卷积等价于频率域的逐元素乘法
- **快速卷积**：利用 FFT 加速大尺寸卷积运算
- **信号滤波**：设计频率响应 $H(\omega)$，滤波即 $Y(\omega)=H(\omega)X(\omega)$

### 10.3 深度学习中的注意力机制

Transformer 中的位置编码利用正弦和余弦函数：

$$PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d}}\right)$$

这种设计的思想源于 Fourier 级数：用不同频率的三角函数编码位置信息，使模型能够区分不同位置。

### 10.4 生成式模型中的频域方法

- **Diffusion Models**：在频域分析噪声添加和去噪过程
- **WaveNet**：利用因果卷积和扩张卷积处理序列
- **Spectral Normalization**：在 GAN 训练中控制判别器的 Lipschitz 常数

### 10.5 信号采样与重建

- **Nyquist-Shannon 采样定理**：采样频率需大于信号最高频率的 2 倍
- **Fourier 变换的离散化**：DFT 和 FFT 是数字信号处理的基础
- **混叠（Aliasing）**：采样不足导致高频信号被误认为低频

---

## 11. 知识脉络

```
数项级数
├── 收敛判别法 ─── 比较、比值、根值、积分
├── 正项级数 ─── p-级数、几何级数
├── 交错级数 ─── Leibniz判别法
└── 绝对/条件收敛 ─── Riemann重排定理

函数项级数
├── 幂级数 ─── 收敛半径、逐项求导/积分
└── Taylor展开 ─── 常见函数展开式、间接展开

Fourier分析
├── Fourier级数 ─── Dirichlet条件、Parseval等式
└── Fourier变换 ─── 卷积定理、性质

Laplace变换
├── 定义与性质 ─── 时域微分、时移
├── 常见变换对
└── 应用 ─── 解微分方程

Z变换
├── 定义 ─── 与Laplace变换的关系
├── 性质 ─── 时移、卷积
└── 应用 ─── 数字滤波器、离散系统分析

应用
├── 信号处理 ─── 频谱分析、FFT、采样定理
├── 深度学习 ─── 位置编码、CNN、注意力机制
└── 生成式模型 ─── 频域方法、Spectral Normalization
```
