# 05-参数估计与假设检验

> 从点估计到区间估计，从检验理论到p值——统计推断的核心方法

---

## 1. 点估计：矩估计法（MM）

### 1.1 基本思想

用样本矩替代总体矩，建立方程组求解参数。

**步骤**：
1. 设总体有 $k$ 个未知参数 $\theta_1, \ldots, \theta_k$
2. 计算前 $k$ 阶总体原点矩 $\mu_j' = E[X^j]$（参数的函数）
3. 令样本矩等于总体矩：$A_j = \mu_j'(\theta_1, \ldots, \theta_k)$，$j=1,\ldots,k$
4. 解方程组得 $\hat{\theta}_1, \ldots, \hat{\theta}_k$

### 1.2 示例

**正态分布** $\mathcal{N}(\mu, \sigma^2)$：

- $E[X] = \mu = A_1 = \bar{X}$
- $E[X^2] = \mu^2 + \sigma^2 = A_2 = \frac{1}{n}\sum X_i^2$

解得：$\hat{\mu} = \bar{X}$，$\hat{\sigma}^2 = A_2 - A_1^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$

### 1.3 优缺点

| 优点 | 缺点 |
|------|------|
| 计算简单，直觉清晰 | 有时效率不高 |
| 不需要似然函数 | 不一定充分利用数据信息 |
| 始终可用 | 可能得到不合理的估计 |

---

## 2. 点估计：最大似然估计法（MLE）

### 2.1 基本思想

选择使观测数据出现概率最大的参数值。

**似然函数**：

$$L(\theta | x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i | \theta)$$

**对数似然**：

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i | \theta)$$

**MLE**：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \ell(\theta)$$

求解：令 $\frac{\partial \ell}{\partial \theta} = 0$（得分方程）。

### 2.2 示例

**指数分布** $f(x) = \lambda e^{-\lambda x}$：

$$\ell(\lambda) = n\ln\lambda - \lambda\sum x_i \implies \hat{\lambda} = \frac{1}{\bar{X}}$$

**正态分布**：

$$\hat{\mu}_{MLE} = \bar{X}, \quad \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(X_i - \bar{X})^2$$

注意 $\hat{\sigma}^2_{MLE}$ 有偏（分母为 $n$ 而非 $n-1$）。

### 2.3 MLE的渐近性质

- **一致性**：$\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$
- **渐近正态性**：$\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} \mathcal{N}(0, \frac{1}{I(\theta_0)})$
- **渐近有效性**：达到Cramer-Rao下界

其中 $I(\theta)$ 是Fisher信息量：

$$I(\theta) = E\left[\left(\frac{\partial \ln f(X|\theta)}{\partial \theta}\right)^2\right] = -E\left[\frac{\partial^2 \ln f(X|\theta)}{\partial \theta^2}\right]$$

### 2.4 与ML的深度关联

- **交叉熵损失** = 负对数似然，最小化交叉熵等价于MLE
- **分类任务**：$L = -\sum \ln P(y_i | x_i, \theta)$ 就是MLE
- **回归任务**：MSE损失等价于假设高斯噪声的MLE
- **正则化**：L2正则化 = 高斯先验的MAP估计

---

## 3. 估计量的评价标准

### 3.1 无偏性（Unbiasedness）

$E[\hat{\theta}] = \theta$，对所有 $\theta$ 成立。

- $\bar{X}$ 是 $\mu$ 的无偏估计
- $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ 是 $\sigma^2$ 的无偏估计（Bessel校正）
- $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(X_i - \bar{X})^2$ 是有偏的

### 3.2 有效性（Efficiency）

在所有无偏估计中，方差最小者最有效。

**Cramer-Rao下界**：任何无偏估计 $\hat{\theta}$ 的方差满足

$$\text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}$$

达到此下界的估计称为**有效估计**。

### 3.3 一致性（Consistency）

$\hat{\theta}_n \xrightarrow{P} \theta$（当 $n \to \infty$）。

- MLE是一致的
- 矩估计是一致的
- 一致性是大样本性质，小样本中可能不成立

### 3.4 均方误差（MSE）

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

有时有偏估计的MSE反而更小（偏差-方差权衡），这正是正则化的理论基础。

---

## 4. 充分统计量与因子分解定理

### 4.1 充分统计量

统计量 $T(X_1, \ldots, X_n)$ 称为 $\theta$ 的充分统计量，如果给定 $T$ 后，样本的条件分布不依赖于 $\theta$。

**直观理解**：$T$ 包含了数据中关于 $\theta$ 的所有信息，不损失任何信息。

### 4.2 因子分解定理

$T$ 是 $\theta$ 的充分统计量 $\Leftrightarrow$ 似然函数可分解为：

$$L(\theta | x_1, \ldots, x_n) = g(T(x_1, \ldots, x_n), \theta) \cdot h(x_1, \ldots, x_n)$$

其中 $g$ 通过 $T$ 依赖数据，$h$ 与 $\theta$ 无关。

### 4.3 常见充分统计量

| 分布 | 充分统计量 |
|------|-----------|
| $\mathcal{N}(\mu, \sigma^2)$（$\sigma^2$已知） | $\bar{X}$ |
| $\mathcal{N}(\mu, \sigma^2)$（均未知） | $(\bar{X}, S^2)$ |
| Binomial($n,p$） | $\sum X_i$ |
| Poisson($\lambda$) | $\sum X_i$ |
| Exponential($\lambda$) | $\sum X_i$ |

---

## 5. 区间估计

### 5.1 置信区间

参数 $\theta$ 的 $1-\alpha$ 置信区间 $[L, U]$ 满足：

$$P(L \leq \theta \leq U) = 1 - \alpha$$

**注意**：置信区间是随机的（依赖样本），参数是固定的。频率学派解释：重复抽样构造的区间中有 $1-\alpha$ 比例包含真值。

### 5.2 正态总体均值的置信区间

**$\sigma^2$ 已知**：$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$

**$\sigma^2$ 未知**：$\bar{X} \pm t_{\alpha/2}(n-1) \cdot \frac{S}{\sqrt{n}}$

其中 $t_{\alpha/2}(\nu)$ 是自由度为 $\nu$ 的 $t$ 分布的上 $\alpha/2$ 分位数。

### 5.3 正态总体方差的置信区间

$$\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)},\ \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right]$$

### 5.4 大样本区间估计

当 $n$ 足够大时（CLT保证），对任意分布：

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{S}{\sqrt{n}}$$

---

## 6. 假设检验

### 6.1 基本概念

- **原假设** $H_0$：默认成立的假设（如无差异、无效果）
- **备择假设** $H_1$：研究者希望证实的假设
- **检验统计量**：用于判断的统计量
- **拒绝域**：使 $H_0$ 被拒绝的统计量取值范围
- **显著性水平** $\alpha$：$H_0$ 为真时被错误拒绝的概率

### 6.2 两类错误

| | $H_0$ 为真 | $H_0$ 为假 |
|--|-----------|-----------|
| 拒绝 $H_0$ | 第I类错误（$\alpha$） | 正确（$1-\beta$，Power） |
| 不拒绝 $H_0$ | 正确（$1-\alpha$） | 第II类错误（$\beta$） |

**Power**（检验功效）$= 1 - \beta = P(\text{拒绝}H_0 | H_1\text{为真})$

### 6.3 检验流程

1. 建立 $H_0$ 和 $H_1$
2. 选择检验统计量及其分布
3. 确定显著性水平 $\alpha$
4. 确定拒绝域
5. 计算统计量的观测值
6. 做出判断

---

## 7. 常用检验方法

### 7.1 Z检验

**适用**：大样本或方差已知的正态总体

检验统计量：$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0,1)$

### 7.2 t检验

**适用**：小样本、方差未知的正态总体

检验统计量：$t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t(n-1)$

**单样本t检验**：检验均值是否等于某值
**双样本t检验**：检验两组均值是否相等（独立样本或配对样本）

### 7.3 卡方检验

**适用**：方差检验、拟合优度检验、列联表独立性检验

检验统计量：$\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1)$

**Pearson卡方检验**（拟合优度）：$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$

### 7.4 F检验

**适用**：两正态总体方差比的检验

检验统计量：$F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)$

**ML应用**：方差分析（ANOVA）用于特征选择和模型比较。

---

## 8. p值

### 8.1 定义

p值是在 $H_0$ 成立的条件下，得到当前观测值或更极端值的概率。

$$p = P(\text{观测到与$H_0$同样极端或更极端的统计量} | H_0\text{为真})$$

### 8.2 判断规则

- $p < \alpha$：拒绝 $H_0$
- $p \geq \alpha$：不拒绝 $H_0$

### 8.3 p值的常见误解

| 错误理解 | 正确理解 |
|----------|----------|
| $p$ 值是 $H_0$ 为真的概率 | $p$ 值是假设 $H_0$ 为真时得到当前数据的概率 |
| $p$ 值是 $H_1$ 为真的概率 | $p$ 值不涉及 $H_1$ 的概率 |
| $p$ 值越大越好 | 取决于研究目的 |

### 8.4 多重检验校正

进行多次检验时，需校正 $\alpha$ 以控制族错误率：

- **Bonferroni校正**：$\alpha' = \alpha / m$
- **FDR控制（Benjamini-Hochberg）**：控制假发现率

**ML应用**：特征选择中筛选显著变量。

---

## 9. 似然比检验

### 9.1 定义

$$\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta)}{\sup_{\theta \in \Theta} L(\theta)}$$

其中 $\Theta_0$ 是 $H_0$ 对应的参数空间，$\Theta$ 是全参数空间。

**Wilks定理**：在正则条件下，

$$-2\ln\Lambda \xrightarrow{d} \chi^2(r)$$

其中 $r = \dim(\Theta) - \dim(\Theta_0)$。

### 9.2 与Wald检验、Score检验的关系

| 检验 | 统计量 | 特点 |
|------|--------|------|
| 似然比检验 | $-2\ln\Lambda$ | 需要拟合两个模型 |
| Wald检验 | $\frac{(\hat{\theta}-\theta_0)^2}{\text{Var}(\hat{\theta})}$ | 只需拟合无约束模型 |
| Score检验 | $\frac{S(\theta_0)^2}{I(\theta_0)}$ | 只需拟合约束模型 |

**ML应用**：模型选择（嵌套模型比较）、逻辑回归中的变量选择。

---

## 参考资料

- 《数理统计学教程》陈希孺
- 《Statistical Inference》Casella & Berger
- 《The Elements of Statistical Learning》Hastie et al.
