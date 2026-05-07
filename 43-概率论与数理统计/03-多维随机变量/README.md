# 03-多维随机变量

> 联合分布、条件分布、独立性、协方差与多元正态分布

---

## 1. 联合分布与边缘分布

### 1.1 联合分布函数

二维随机变量 $(X,Y)$ 的联合分布函数：

$$F(x,y) = P(X \leq x, Y \leq y)$$

**性质**：
- 对每个变量单调不减
- $F(+\infty, +\infty) = 1$，$F(-\infty, y) = F(x, -\infty) = 0$
- 右连续

### 1.2 联合概率密度

**离散型**：联合分布律 $P(X=x_i, Y=y_j) = p_{ij}$

**连续型**：联合PDF $f(x,y)$ 满足

$$F(x,y) = \int_{-\infty}^{x}\int_{-\infty}^{y} f(u,v)\,dv\,du$$

$$P((X,Y) \in D) = \iint_D f(x,y)\,dx\,dy$$

### 1.3 边缘分布

从联合分布中"去掉"一个变量的分布。

**离散型**：
$$P(X = x_i) = \sum_j p_{ij}, \quad P(Y = y_j) = \sum_i p_{ij}$$

**连续型**：
$$f_X(x) = \int_{-\infty}^{+\infty} f(x,y)\,dy, \quad f_Y(y) = \int_{-\infty}^{+\infty} f(x,y)\,dx$$

**关键认识**：联合分布确定边缘分布，但边缘分布**不能**唯一确定联合分布（丢失了变量间的关联信息）。

**ML关联**：这正是概率图模型的核心——通过联合分布建模变量间依赖关系。

---

## 2. 条件分布

### 2.1 离散型条件分布

$$P(X = x_i | Y = y_j) = \frac{P(X=x_i, Y=y_j)}{P(Y=y_j)} = \frac{p_{ij}}{p_{\cdot j}}$$

### 2.2 连续型条件分布

**条件PDF**：

$$f_{X|Y}(x|y) = \frac{f(x,y)}{f_Y(y)}, \quad f_Y(y) > 0$$

$$f_{Y|X}(y|x) = \frac{f(x,y)}{f_X(x)}, \quad f_X(x) > 0$$

### 2.3 贝叶斯公式的连续形式

$$f_{Y|X}(y|x) = \frac{f_{X|Y}(x|y) \cdot f_Y(y)}{f_X(x)}$$

这正是贝叶斯推断的核心公式：先验 $f_Y(y)$ 通过似然 $f_{X|Y}(x|y)$ 更新为后验 $f_{Y|X}(y|x)$。

---

## 3. 随机变量的独立性

### 3.1 定义

$(X,Y)$ 独立 $\Leftrightarrow$ 对所有 $x,y$：

$$F(x,y) = F_X(x) \cdot F_Y(y)$$

等价条件：

| 类型 | 独立的充要条件 |
|------|---------------|
| 离散 | $P(X=x_i, Y=y_j) = P(X=x_i) \cdot P(Y=y_j)$，对所有 $i,j$ |
| 连续 | $f(x,y) = f_X(x) \cdot f_Y(y)$，对几乎所有 $(x,y)$ |

### 3.2 独立性的判断

- **从定义出发**：检验联合密度是否可分解为边缘密度的乘积
- **支撑集检验**：若联合PDF的支撑集不是矩形区域（可分解的），则不独立
- **独立时**：$E[XY] = E[X]E[Y]$，$\text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y)$

---

## 4. 协方差与相关系数

### 4.1 协方差

$$\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]$$

**性质**：
- $\text{Cov}(X,X) = \text{Var}(X)$
- $\text{Cov}(X,Y) = \text{Cov}(Y,X)$
- $\text{Cov}(aX+b,\ cY+d) = ac \cdot \text{Cov}(X,Y)$
- $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$
- $\text{Cov}(X+Y, Z) = \text{Cov}(X,Z) + \text{Cov}(Y,Z)$（双线性）

### 4.2 相关系数

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \cdot \sigma_Y}$$

- $\rho \in [-1, 1]$
- $|\rho| = 1$ 当且仅当 $X$ 与 $Y$ 线性相关（$Y = aX+b$ 几乎处处成立）
- $\rho = 0$ 称为**不相关**，但不相关 $\not\Rightarrow$ 独立

### 4.3 不相关 vs 独立

| | 独立 | 不相关 |
|--|------|--------|
| 定义 | $f(x,y)=f_X(x)f_Y(y)$ | $\rho=0$（即 $\text{Cov}=0$） |
| 关系 | 独立 $\Rightarrow$ 不相关 | 不相关 $\not\Rightarrow$ 独立 |
| 反例 | — | $X \sim \mathcal{N}(0,1)$，$Y=X^2$，$\text{Cov}=0$ 但不独立 |

**特例**：对于**二元正态分布**，不相关等价于独立。

**ML关联**：特征工程中需要去除相关性（PCA的核心思想），相关系数矩阵即协方差矩阵的标准化形式。

---

## 5. 多维正态分布

### 5.1 二元正态分布 $\mathcal{N}(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$

$$f(x,y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2} + \frac{(y-\mu_2)^2}{\sigma_2^2}\right]\right)$$

**关键性质**：
- 边缘分布均为正态
- 条件分布也是正态
- $\rho = 0$ $\Leftrightarrow$ $X, Y$ 独立

### 5.2 多元正态分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

- $\boldsymbol{\mu} \in \mathbb{R}^d$：均值向量
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$：协方差矩阵（对称半正定）

**重要性质**：
- 线性变换仍为正态：$A\mathbf{X}+\mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b},\ A\boldsymbol{\Sigma}A^T)$
- 边缘分布和条件分布均为正态

**ML应用**：高斯过程回归、概率PCA、变分自编码器（VAE）的后验近似、Diffusion Model。

---

## 6. 随机变量函数的分布

### 6.1 一般变换法

设 $Y = g(X)$，$g$ 单调可导：

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$$

### 6.2 卷积公式

$Z = X + Y$，$X, Y$ 独立：

$$f_Z(z) = \int_{-\infty}^{+\infty} f_X(x) f_Y(z-x)\,dx = (f_X * f_Y)(z)$$

**常见结果**：
- 正态 + 正态 = 正态：$\mathcal{N}(\mu_1,\sigma_1^2) + \mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$
- Gamma + Gamma（同参数）= Gamma
- 二项分布的可加性

### 6.3 常用变换公式

**$Z = X/Y$**（$X, Y$ 独立）：

$$f_Z(z) = \int_{-\infty}^{+\infty} |y|\, f_X(zy)\, f_Y(y)\,dy$$

**$Z = \max(X,Y)$**（$X, Y$ 独立）：

$$F_Z(z) = F_X(z) \cdot F_Y(z)$$

**$Z = \min(X,Y)$**（$X, Y$ 独立）：

$$F_Z(z) = 1 - (1-F_X(z))(1-F_Y(z))$$

推广到 $n$ 个独立变量 $X_1, \ldots, X_n$：

$$F_{\max}(z) = \prod_{i=1}^n F_{X_i}(z), \quad F_{\min}(z) = 1 - \prod_{i=1}^n (1-F_{X_i}(z))$$

---

## 7. 次序统计量

### 7.1 定义

设 $X_1, X_2, \ldots, X_n$ 独立同分布（i.i.d.），将其排序：

$$X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$$

$X_{(k)}$ 称为第 $k$ 个次序统计量。

- $X_{(1)} = \min(X_1, \ldots, X_n)$：最小值
- $X_{(n)} = \max(X_1, \ldots, X_n)$：最大值
- $X_{(\lceil n/2 \rceil)}$：样本中位数

### 7.2 次序统计量的分布

设总体CDF为 $F(x)$，PDF为 $f(x)$。

**单个次序统计量** $X_{(k)}$ 的PDF：

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F(x)]^{k-1}[1-F(x)]^{n-k} f(x)$$

**极值分布**：
$$f_{X_{(n)}}(x) = n[F(x)]^{n-1}f(x)$$
$$f_{X_{(1)}}(x) = n[1-F(x)]^{n-1}f(x)$$

### 7.3 应用

- **稳健统计**：中位数比均值对异常值更鲁棒
- **极值理论**：金融风险管理、洪水频率分析
- **ML应用**：排序学习（Learning to Rank）、Top-K选择

---

## 参考资料

- 《概率论与数理统计》陈希孺
- 《多元统计分析》方开泰
- 《Pattern Recognition and Machine Learning》Bishop, Ch.2
