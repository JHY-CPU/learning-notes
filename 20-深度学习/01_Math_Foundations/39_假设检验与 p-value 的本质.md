# 39_假设检验与 p-value 的本质

## 核心概念

- **假设检验 (Hypothesis Testing)**：统计推断的核心工具，用于判断数据是否支持某个假设。设立两个对立的假设：零假设 $H_0$（通常表示"无效果"或"无差异"）和备择假设 $H_1$。
- **p-value 的定义**：p-value 是在零假设 $H_0$ 为真的前提下，观测到当前结果（或更极端结果）的概率。$\text{p-value} = P(\text{数据}|H_0)$。**不是** $P(H_0|\text{数据})$！
- **第一类与第二类错误**：
  - 第一类错误（Type I）：错误地拒绝 $H_0$（假阳性），概率 $\alpha$（显著性水平）
  - 第二类错误（Type II）：错误地接受 $H_0$（假阴性），概率 $\beta$
  - 统计功效 (Power)：$1 - \beta$，即正确拒绝 $H_0$ 的概率
- **显著性水平 $\alpha$**：预设的阈值（通常 0.05），p-value < $\alpha$ 时拒绝 $H_0$。这并不意味着"95% 的概率 $H_0$ 是错的"。
- **p-value 的常见误解**：
  - 不是 $H_0$ 为真的概率
  - 不是效应量
  - 不是 $H_0$ 为假时结论可靠的概率
  - 不是实验重复时得到同样结果的概率
- **效应量 (Effect Size)**：p-value 受样本量影响巨大，大样本下微小差异也可能显著。效应量（如 Cohen's d、相关系数）衡量差异的实际大小，不受样本量影响。

## 数学推导

假设检验的一般流程：
- 设定 $H_0$ 和 $H_1$
- 选择检验统计量 $T(X)$
- 计算在 $H_0$ 下 $T(X)$ 的分布
- 计算 p-value = $P_{H_0}(T(X) \geq T_{\text{obs}})$
- 若 p-value < $\alpha$，拒绝 $H_0$

双样本 t 检验示例：
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{1/n_1 + 1/n_2}}
$$

其中 $s_p$ 是合并标准差：
$$
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
$$

p-value = $P(|T| \geq |t|)$，其中 $T \sim t_{n_1+n_2-2}$（t 分布）。

多重检验校正（Bonferroni）：
$$
\alpha_{\text{adjusted}} = \frac{\alpha}{m}
$$
其中 $m$ 是同时进行的检验次数。

## 直观理解

- **"在法庭上证明有罪"的类比**：零假设 $H_0$ = "被告无罪"，证据 = 观测数据。p-value 是"假设被告无罪，观察到当前证据（或更强证据）的概率"。p-value 很小（< 0.05），说明如果被告无罪，当前证据几乎不可能出现，于是"拒绝无罪假设"。
- **重要：p-value 不是 $H_0$ 成立的概率**：即使 p-value = 0.03，也不意味着 $H_0$ 只有 3% 的概率成立。p-value 说的是"在 $H_0$ 下数据的概率"，而不是"在数据下 $H_0$ 的概率"。后者需要贝叶斯方法（贝叶斯因子）。
- **p-value 的"非对称"性**：p-value 只能提供拒绝 $H_0$ 的证据，不能提供接受 $H_0$ 的证据。p-value > 0.05 只能说"没有足够证据拒绝 $H_0$"，不能说"$H_0$ 为真"。

## 代码示例

```python
import numpy as np
from scipy import stats

# 1. 单样本 t 检验
np.random.seed(42)

# 生成数据：均值 = 0.3，检验 H0: 均值 = 0
n = 50
data = np.random.randn(n) + 0.3

t_stat, p_value = stats.ttest_1samp(data, 0)
print("单样本 t 检验 (H0: μ=0):")
print(f"  数据均值={np.mean(data):.4f}")
print(f"  t统计量={t_stat:.4f}, p-value={p_value:.4f}")
print(f"  结论: {'拒绝 H0' if p_value < 0.05 else '不能拒绝 H0'}")

# 2. 双样本 t 检验
group1 = np.random.randn(30) + 0.5
group2 = np.random.randn(30) + 0.0

t_stat2, p_value2 = stats.ttest_ind(group1, group2)
print(f"\n双样本 t 检验 (H0: μ1=μ2):")
print(f"  组1均值={np.mean(group1):.4f}, 组2均值={np.mean(group2):.4f}")
print(f"  t统计量={t_stat2:.4f}, p-value={p_value2:.4f}")

# 3. p-value 的分布与样本量
print("\n样本量对 p-value 的影响 (真实差异=0.2):")
for n in [10, 50, 200, 1000]:
    p_vals = []
    for _ in range(1000):
        x = np.random.randn(n) + 0.2
        _, p = stats.ttest_1samp(x, 0)
        p_vals.append(p)
    sig_ratio = np.mean([p < 0.05 for p in p_vals])
    print(f"  n={n:4d}: 显著比例={sig_ratio:.3f} (统计功效)")

# 4. p-value 的分布：H0 为真时
print("\nH0 为真时 p-value 的分布 (均匀!):")
p_vals_null = []
for _ in range(5000):
    x = np.random.randn(100)  # 均值=0
    _, p = stats.ttest_1samp(x, 0)
    p_vals_null.append(p)
print(f"  p<0.05 比例: {np.mean([p < 0.05 for p in p_vals_null]):.4f} (应≈0.05)")

# 5. 多重检验校正
print("\n多重检验校正 (Bonferroni):")
n_tests = 20
p_values = np.array([
    0.001, 0.01, 0.03, 0.04, 0.045,  # 显著
    0.06, 0.1, 0.2, 0.3, 0.4,         # 不显著
    0.5, 0.6, 0.7, 0.8, 0.9,          # 不显著
    0.002, 0.02, 0.05, 0.08, 0.15      # 混合
])

alpha = 0.05
bonf_alpha = alpha / n_tests
significant_raw = p_values < alpha
significant_bonf = p_values < bonf_alpha

print(f"  检验次数: {n_tests}")
print(f"  原始 α = {alpha}, Bonferroni α = {bonf_alpha:.6f}")
print(f"  原始发现显著: {np.sum(significant_raw)}")
print(f"  Bonferroni发现显著: {np.sum(significant_bonf)}")

# 6. 效应量 vs p-value
print("\n效应量与 p-value 的对比:")
for effect in [0.1, 0.3, 0.5, 0.8]:
    for n in [20, 200]:
        p_vals = []
        for _ in range(100):
            x = np.random.randn(n) + effect
            _, p = stats.ttest_1samp(x, 0)
            p_vals.append(p)
        mean_p = np.mean(p_vals)
        cohens_d = effect  # 近似 d = (μ-0)/σ
        print(f"  d={effect:.1f}, n={n:3d}: 平均p={mean_p:.4f}")
```

## 深度学习关联

- **模型比较中的假设检验**：在深度学习研究中，假设检验用于比较不同模型的性能。McNemar 检验或配对 t 检验可以判断模型 A 和模型 B 在测试集上的精度差异是否统计显著。但需要注意，大测试集下微小差异也会显著，应结合效应量（如 Cohen's d）解释。
- **Ablation Study 与显著性**：消融实验（Ablation Study）的目的是检验某个组件对模型性能的贡献。p-value 可以帮助判断"观测到的性能下降是随机波动还是组件确实重要"。
- **p-hacking 与深度学习危机**：深度学习领域的"p-hacking"问题——研究者可能多次调参、更换随机种子，直到得到显著结果。这导致了可重复性危机。解决方法是预先注册实验计划、使用 hold-out 测试集、报告置信区间而非单一 p-value。
- **贝叶斯替代：贝叶斯因子 (Bayes Factor)**：相比于 p-value，贝叶斯因子 $BF_{10} = P(\text{数据}|H_1)/P(\text{数据}|H_0)$ 可以直接比较 H0 和 H1 的相对证据。在深度学习中，贝叶斯方法（如贝叶斯假设检验）提供更直观的不确定性量化，在医疗诊断等高风险应用中日益重要。
