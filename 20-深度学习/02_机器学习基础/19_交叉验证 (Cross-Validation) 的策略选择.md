# 20_交叉验证 (Cross-Validation) 的策略选择

## 核心概念

- **交叉验证**：一种评估模型泛化能力的方法，通过将数据集划分为互补的子集，在部分子集上训练、在剩余子集上验证，多次重复后取平均。
- **K 折交叉验证 (K-Fold CV)**：将数据等分为 $K$ 份，每次取 $K-1$ 份训练、1 份验证，循环 $K$ 次，取平均性能。$K$ 通常取 5 或 10。
- **留一法 (LOOCV)**：$K = m$（样本数），每次留一个样本做验证。偏差最小但方差大且计算成本极高，适合小样本。
- **分层交叉验证 (Stratified K-Fold)**：在划分时保持每折中类别分布与原始数据一致，对不平衡分类任务尤为重要。
- **重复交叉验证 (Repeated K-Fold)**：多次运行 $K$ 折交叉验证（每次随机打乱），进一步降低评估的方差，提供更稳定的性能估计。
- **时间序列交叉验证**：针对时间序列数据，使用时间顺序划分——训练集只能包含验证集之前的数据，避免未来信息泄露。

## 数学推导

**K 折交叉验证的误差估计**：

设数据 $D$ 被划分为 $K$ 个子集 $D_1, D_2, \dots, D_K$，每个子集大小约为 $m/K$。第 $k$ 折的模型 $\hat{f}_k$ 在 $D \setminus D_k$ 上训练，在 $D_k$ 上评估。

交叉验证误差为：
$$
CV(\hat{f}) = \frac{1}{K} \sum_{k=1}^K \frac{1}{|D_k|} \sum_{i \in D_k} L(y_i, \hat{f}_k(x_i))
$$

**偏差和方差分析**：
- LOOCV 的偏差最小，因为使用了几乎全部数据训练，估计近似无偏
- 但 LOOCV 的方差可能很大，因为 $m$ 个高度重叠的训练集导致估计之间高度相关
- 5 折或 10 折 CV 在偏差和方差之间取得较好平衡

**CV 估计的性质**：
$$
\mathbb{E}[CV(\hat{f})] = \text{泛化误差} + \text{小偏差项}
$$

经验研究表明，$K=5$ 或 $K=10$ 时 CV 估计的方差最小，偏差也处于可接受水平。对于分类问题，分层 CV 通常优于普通 CV。

**单次训练-验证划分 vs CV**：单次划分（如 70/30）的方差远大于 CV，因为验证集小且对划分方式敏感。CV 通过平均多个划分来降低方差。

## 直观理解

- **"交叉验证 = 反复考试"**：单次训练-验证划分像一次期末考试，偶然因素大（考试当天状态、试卷覆盖面等）。交叉验证像每月一次小测验——10 次考试的平均成绩比 1 次期末成绩更能反映真实水平。
- **K 值的选择"中庸之道"**：$K$ 太大（如 LOOCV），相当于每次只留一道题做测试，虽然利用了几乎所有题目来学习，但每次测试的结果波动大（成绩可能从 90 到 95 波动）。$K$ 太小（如 2 折），每次只用了 50% 的数据训练，模型学得不够好，偏差大。$K=5$ 或 $10$ 是经验上的甜点。
- **分层的重要性**：如果测试卷中 90% 是选择题、10% 是简答题，而每次只考一个题型，成绩就不能反映真实水平。分层 CV 保证每次考试的类型分布与总分布一致，评估更公平。

## 代码示例

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     RepeatedKFold, cross_val_score,
                                     TimeSeriesSplit)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

rf = RandomForestClassifier(n_estimators=50, random_state=42)

# 1. 标准 K 折
scores_k5 = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"5折 CV: {scores_k5.mean():.4f} +/- {scores_k5.std():.4f}")

# 2. 分层 K 折（分类任务推荐）
stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(rf, X, y, cv=stratified, scoring='accuracy')
print(f"分层5折 CV: {scores_strat.mean():.4f} +/- {scores_strat.std():.4f}")

# 3. 重复 K 折
repeated = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores_rep = cross_val_score(rf, X, y, cv=repeated, scoring='accuracy')
print(f"重复5×10 CV: {scores_rep.mean():.4f} +/- {scores_rep.std():.4f}")

# 4. 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
X_ts = np.arange(100).reshape(-1, 1)
y_ts = np.sin(X_ts.ravel()) + np.random.randn(100) * 0.1
scores_ts = cross_val_score(RandomForestClassifier(), X_ts[:50], y_ts[:50] > 0,
                            cv=tscv, scoring='accuracy')
print(f"时间序列 CV: {scores_ts}")
```

## 深度学习关联

- **深度学习的 CV 挑战**：深度模型训练成本极高，全 K 折 CV 的代价过大。实践中通常采用单次验证集划分，或用轻量级方法如 Early Stopping 的验证损失曲线来替代 CV。
- **微调中的 CV**：在预训练-微调范式中，微调阶段数据量通常较小，使用 5 折 CV 来评估不同微调策略和超参数是推荐做法，能更可靠地选择最优配置。
- **嵌套交叉验证**：在深度学习的架构搜索 (NAS) 中，嵌套 CV（外层评估架构、内层调参）是避免过拟合搜索过程的标准方法，与 AutoML 框架（如 Optuna、Hyperopt）结合使用。
