# 04-多元微积分

> 从一元到多元，数学分析从线拓展到面和空间。本章涵盖多元函数的微分学（偏导数、方向导数、梯度）与积分学（重积分、曲线曲面积分），以及场论三大公式，并讨论在机器学习中的应用。

## 1. 多元函数的极限与连续

### 1.1 多元函数的概念

二元函数 $z=f(x,y)$ 的定义域是 $xy$ 平面上的区域。类似地推广到 $n$ 元函数 $f(x_1,x_2,\ldots,x_n)$。

### 1.2 二重极限

$$\lim_{(x,y)\to(x_0,y_0)}f(x,y)=A$$

$\forall\varepsilon>0$，$\exists\delta>0$，当 $0<\sqrt{(x-x_0)^2+(y-y_0)^2}<\delta$ 时，$|f(x,y)-A|<\varepsilon$。

**关键点**：$(x,y)$ 沿任何路径趋近 $(x_0,y_0)$ 时极限必须相同。若不同路径极限不同，则二重极限不存在。

### 1.3 累次极限与二重极限的区别

- **累次极限**：$\lim_{x\to x_0}\lim_{y\to y_0}f(x,y)$ 与 $\lim_{y\to y_0}\lim_{x\to x_0}f(x,y)$
- 二重极限存在不能推出累次极限存在，反之亦然
- 若二重极限和两个累次极限都存在，则三者相等

### 1.4 连续性

若 $\lim_{(x,y)\to(x_0,y_0)}f(x,y)=f(x_0,y_0)$，则 $f(x,y)$ 在 $(x_0,y_0)$ 处连续。

连续函数的四则运算和复合仍连续。有界闭区域（紧集）上连续函数具有有界性、最值定理、介值定理。

---

## 2. 偏导数与全微分

### 2.1 偏导数

$$\frac{\partial f}{\partial x}\bigg|_{(x_0,y_0)}=\lim_{\Delta x\to 0}\frac{f(x_0+\Delta x,y_0)-f(x_0,y_0)}{\Delta x}$$

$\frac{\partial f}{\partial x}$ 本质上是将 $y$ 固定在 $y_0$，对 $x$ 求导。

### 2.2 高阶偏导数

$$f_{xx}=\frac{\partial^2 f}{\partial x^2},\quad f_{xy}=\frac{\partial^2 f}{\partial y\partial x},\quad f_{yx}=\frac{\partial^2 f}{\partial x\partial y},\quad f_{yy}=\frac{\partial^2 f}{\partial y^2}$$

**定理**：若 $f_{xy}$ 和 $f_{yx}$ 连续，则 $f_{xy}=f_{yx}$（混合偏导数与求导顺序无关）。

### 2.3 全微分

若 $\Delta z=f(x+\Delta x,y+\Delta y)-f(x,y)=A\Delta x+B\Delta y+o(\rho)$，其中 $\rho=\sqrt{(\Delta x)^2+(\Delta y)^2}$，则：

$$dz=\frac{\partial f}{\partial x}dx+\frac{\partial f}{\partial y}dy$$

- 可微 $\Rightarrow$ 连续
- 可微 $\Rightarrow$ 偏导数存在
- 偏导数存在 $\nRightarrow$ 可微
- **可微的充分条件**：偏导数连续 $\Rightarrow$ 可微

### 2.4 全微分的应用

近似计算：$f(x+\Delta x,y+\Delta y)\approx f(x,y)+f_x\Delta x+f_y\Delta y$。

误差估计：$|\Delta z|\approx|dz|\leq|f_x||\Delta x|+|f_y||\Delta y|$。

---

## 3. 多元复合函数求导（链式法则）

### 3.1 基本情形

设 $z=f(u,v)$，$u=\varphi(x,y)$，$v=\psi(x,y)$，则：

$$\frac{\partial z}{\partial x}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial x}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial x}$$

$$\frac{\partial z}{\partial y}=\frac{\partial z}{\partial u}\frac{\partial u}{\partial y}+\frac{\partial z}{\partial v}\frac{\partial v}{\partial y}$$

### 3.2 特殊情形

- **全导数**：$z=f(u,v)$，$u=\varphi(t)$，$v=\psi(t)$，则 $\frac{dz}{dt}=\frac{\partial f}{\partial u}\frac{du}{dt}+\frac{\partial f}{\partial v}\frac{dv}{dt}$
- **混合情形**：$z=f(x,y,t)$，$x=\varphi(s,t)$，$y=\psi(s,t)$，则 $\frac{\partial z}{\partial t}=\frac{\partial f}{\partial x}\frac{\partial x}{\partial t}+\frac{\partial f}{\partial y}\frac{\partial y}{\partial t}+\frac{\partial f}{\partial t}$

### 3.3 链式法则的记忆方法

画出变量依赖关系的树状图，从因变量到自变量的每条路径贡献一个乘积项，各路径求和。

---

## 4. 隐函数求导

### 4.1 一个方程的情形

**一元隐函数**：$F(x,y)=0$ 确定 $y=y(x)$，则：

$$\frac{dy}{dx}=-\frac{F_x}{F_y}\quad(F_y\neq 0)$$

**二元隐函数**：$F(x,y,z)=0$ 确定 $z=z(x,y)$，则：

$$\frac{\partial z}{\partial x}=-\frac{F_x}{F_z},\quad\frac{\partial z}{\partial y}=-\frac{F_y}{F_z}\quad(F_z\neq 0)$$

### 4.2 方程组的情形

设 $\begin{cases}F(x,y,u,v)=0\\G(x,y,u,v)=0\end{cases}$ 确定 $u=u(x,y)$，$v=v(x,y)$。

利用 Jacobi 行列式：

$$J=\frac{\partial(F,G)}{\partial(u,v)}=\begin{vmatrix}F_u&F_v\\G_u&G_v\end{vmatrix}$$

$$\frac{\partial u}{\partial x}=-\frac{1}{J}\frac{\partial(F,G)}{\partial(x,v)},\quad\frac{\partial v}{\partial x}=-\frac{1}{J}\frac{\partial(F,G)}{\partial(u,x)}$$

---

## 5. 方向导数与梯度

### 5.1 方向导数

设 $\mathbf{l}=(\cos\alpha,\cos\beta)$ 为方向余弦表示的方向，则：

$$\frac{\partial f}{\partial \mathbf{l}}=\lim_{t\to 0^+}\frac{f(x_0+t\cos\alpha,y_0+t\cos\beta)-f(x_0,y_0)}{t}$$

若 $f$ 可微，则：

$$\frac{\partial f}{\partial \mathbf{l}}=\frac{\partial f}{\partial x}\cos\alpha+\frac{\partial f}{\partial y}\cos\beta$$

### 5.2 梯度

$$\nabla f=\text{grad}\,f=\left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\right)$$

推广到 $n$ 维：$\nabla f=\left(\frac{\partial f}{\partial x_1},\ldots,\frac{\partial f}{\partial x_n}\right)$。

### 5.3 梯度的性质

- **方向导数与梯度的关系**：$\frac{\partial f}{\partial \mathbf{l}}=\nabla f\cdot\mathbf{l}=|\nabla f|\cos\theta$
- **梯度方向**是函数增长最快的方向，最大增长率为 $|\nabla f|$
- **梯度与等值线/等值面**垂直
- $\nabla f$ 指向 $f$ 增大的方向

---

## 6. 多元函数极值

### 6.1 极值的必要条件

若 $f(x,y)$ 在 $(x_0,y_0)$ 处取极值且偏导数存在，则：

$$f_x(x_0,y_0)=0,\quad f_y(x_0,y_0)=0$$

满足此条件的点称为驻点（或临界点）。

### 6.2 极值的充分条件（Hessian判别）

令 $A=f_{xx}(x_0,y_0)$，$B=f_{xy}(x_0,y_0)$，$C=f_{yy}(x_0,y_0)$，$\Delta=AC-B^2$：

| 条件 | 结论 |
|---|---|
| $\Delta>0$，$A>0$ | 极小值 |
| $\Delta>0$，$A<0$ | 极大值 |
| $\Delta<0$ | 非极值（鞍点） |
| $\Delta=0$ | 无法判断，需进一步分析 |

### 6.3 Hessian矩阵

$$H=\begin{pmatrix}f_{xx}&f_{xy}\\f_{yx}&f_{yy}\end{pmatrix}$$

推广到 $n$ 维：$H_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}$。

- $H$ 正定 $\Rightarrow$ 极小值
- $H$ 负定 $\Rightarrow$ 极大值
- $H$ 不定 $\Rightarrow$ 鞍点

---

## 7. 条件极值与Lagrange乘数法

### 7.1 问题形式

在约束 $\varphi(x,y)=0$ 下求 $f(x,y)$ 的极值。

### 7.2 Lagrange乘数法

构造 Lagrange 函数：

$$\mathcal{L}(x,y,\lambda)=f(x,y)+\lambda\varphi(x,y)$$

解方程组：

$$\begin{cases}\mathcal{L}_x=f_x+\lambda\varphi_x=0\\\mathcal{L}_y=f_y+\lambda\varphi_y=0\\\mathcal{L}_\lambda=\varphi(x,y)=0\end{cases}$$

### 7.3 多约束情形

在约束 $\varphi(x,y,z)=0$ 和 $\psi(x,y,z)=0$ 下：

$$\mathcal{L}(x,y,z,\lambda,\mu)=f(x,y,z)+\lambda\varphi(x,y,z)+\mu\psi(x,y,z)$$

### 7.4 应用

- 最短距离、最优投资组合
- 机器学习中的正则化（约束优化）
- SVM 中的最大间隔分类器

---

## 8. 重积分

### 8.1 二重积分

$$\iint_D f(x,y)\,dA=\lim_{\lambda\to 0}\sum_{k=1}^{n}f(\xi_k,\eta_k)\Delta A_k$$

### 8.2 Fubini定理（累次积分）

若 $D$ 为 $a\leq x\leq b$，$\varphi_1(x)\leq y\leq\varphi_2(x)$，则：

$$\iint_D f\,dA=\int_a^b\left[\int_{\varphi_1(x)}^{\varphi_2(x)}f(x,y)\,dy\right]dx$$

**换序技巧**：画出积分区域，改变积分次序有时可大幅简化计算。

### 8.3 极坐标变换

$$\iint_D f(x,y)\,dA=\iint_{D'}f(r\cos\theta,r\sin\theta)\,r\,dr\,d\theta$$

额外因子 $r$ 来自 Jacobi 行列式 $\left|\frac{\partial(x,y)}{\partial(r,\theta)}\right|=r$。

### 8.4 三重积分

$$\iiint_\Omega f(x,y,z)\,dV$$

常用坐标变换：
- **柱坐标**：$x=r\cos\theta$，$y=r\sin\theta$，$z=z$，$dV=r\,dr\,d\theta\,dz$
- **球坐标**：$x=\rho\sin\varphi\cos\theta$，$y=\rho\sin\varphi\sin\theta$，$z=\rho\cos\varphi$，$dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$

---

## 9. 曲线积分与曲面积分

### 9.1 第一类曲线积分（标量场沿曲线的积分）

$$\int_L f(x,y)\,ds=\int_\alpha^\beta f[x(t),y(t)]\sqrt{x'(t)^2+y'(t)^2}\,dt$$

物理意义：曲线形构件的质量。

### 9.2 第二类曲线积分（向量场沿曲线的积分）

$$\int_L \mathbf{F}\cdot d\mathbf{r}=\int_L P\,dx+Q\,dy=\int_\alpha^\beta[P\,x'(t)+Q\,y'(t)]\,dt$$

物理意义：变力沿曲线做功。

### 9.3 第一类曲面积分

$$\iint_S f(x,y,z)\,dS=\iint_D f[x,y,z(x,y)]\sqrt{1+z_x^2+z_y^2}\,dA$$

### 9.4 第二类曲面积分（通量积分）

$$\iint_S \mathbf{F}\cdot d\mathbf{S}=\iint_S(P\cos\alpha+Q\cos\beta+R\cos\gamma)\,dS$$

其中 $(\cos\alpha,\cos\beta,\cos\gamma)$ 为 $S$ 的单位法向量。

---

## 10. Green公式、Stokes公式、Gauss公式

### 10.1 Green公式

平面闭区域 $D$ 的边界 $L$（正向）：

$$\oint_L P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA$$

**应用**：
- 计算平面面积：$A=\frac{1}{2}\oint_L x\,dy-y\,dx$
- 判断曲线积分与路径无关：$\frac{\partial Q}{\partial x}=\frac{\partial P}{\partial y}$

### 10.2 Stokes公式

$$\oint_L \mathbf{F}\cdot d\mathbf{r}=\iint_S(\nabla\times\mathbf{F})\cdot d\mathbf{S}$$

即：沿闭曲线的环量等于旋度通过曲面的通量。Green 公式是 Stokes 公式在 $xy$ 平面上的特例。

### 10.3 Gauss公式（散度定理）

$$\oiint_S \mathbf{F}\cdot d\mathbf{S}=\iiint_\Omega(\nabla\cdot\mathbf{F})\,dV$$

即：通过闭曲面的通量等于散度在区域内的体积分。

### 10.4 三大公式的统一

三者都是微积分基本定理在高维的推广：
- Green/Stokes：边界的积分 = 内部的微分
- Gauss：边界的积分 = 内部的散度积分

**场论三大公式的关系**：
```
Newton-Leibniz ──→ Green（平面）──→ Stokes（曲面）──→ Gauss（空间）
一元微积分基本定理    平面场论         旋度定理          散度定理
```

---

## 11. 在机器学习中的应用

### 11.1 梯度下降

梯度下降是最核心的优化算法：

$$\mathbf{x}_{k+1}=\mathbf{x}_k-\eta\nabla f(\mathbf{x}_k)$$

- 沿梯度的负方向移动，每次步长为 $\eta$（学习率）
- 目标函数 $f(\mathbf{x})$ 沿梯度反方向下降最快
- 变体：随机梯度下降（SGD）、小批量梯度下降、Adam、RMSProp

### 11.2 凸优化基础

**凸函数**：$f(\lambda x+(1-\lambda)y)\leq\lambda f(x)+(1-\lambda)f(y)$

- 凸函数的局部极小值就是全局极小值
- $f$ 可微且凸 $\Leftrightarrow$ $f(y)\geq f(x)+\nabla f(x)^T(y-x)$
- $f$ 二次可微且凸 $\Leftrightarrow$ Hessian 矩阵 $H\succeq 0$（半正定）

### 11.3 Jacobian与Hessian

- **Jacobian 矩阵**：向量值函数 $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$ 的导数 $J_{ij}=\frac{\partial f_i}{\partial x_j}$
- **Hessian 矩阵**：标量函数的二阶导数，用于判断极值类型和 Newton 法
- 在深度学习中：反向传播本质上是链式法则在多层复合函数上的应用

### 11.4 Lagrange乘数法与约束优化

- SVM 的最大间隔问题：$\min\frac{1}{2}\|w\|^2$，约束 $y_i(w^Tx_i+b)\geq 1$
- 正则化可以理解为约束优化：L2 正则化约束参数在球内，L1 正则化约束参数在菱形内

---

## 12. 知识脉络

```
多元微分学
├── 极限与连续 ─── 二重极限、路径无关性
├── 偏导数与全微分 ─── 可微条件、链式法则
├── 隐函数求导 ─── Jacobi行列式
├── 方向导数与梯度 ─── 最速下降方向
├── 极值 ─── Hessian判别、Lagrange乘数法

多元积分学
├── 重积分 ─── 二重/三重、Fubini定理、坐标变换
├── 曲线积分 ─── 第一类（标量）、第二类（向量）
├── 曲面积分 ─── 第一类（标量）、第二类（通量）
└── 场论 ─── Green、Stokes、Gauss公式

机器学习应用
├── 梯度下降 ─── SGD、Adam
├── 凸优化 ─── 全局最优性
└── 约束优化 ─── SVM、正则化
```
