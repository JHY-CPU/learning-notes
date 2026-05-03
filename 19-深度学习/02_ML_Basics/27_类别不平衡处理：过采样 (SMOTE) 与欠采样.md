# 27_类别不平衡处理：过采样 (SMOTE) 与欠采样

## 核心概念

- **类别不平衡**：分类任务中不同类别的样本数量差异悬殊（如欺诈检测中正类仅占 0.1%），导致模型偏向多数类，对少数类的预测能力差。
- **欠采样 (Under-sampling)**：随机从多数类中移除样本，使各类别数量接近。简单但可能丢失多数类的有用信息。
- **过采样 (Over-sampling)**：随机复制少数类样本或生成合成样本。简单复制可能导致过拟合。
- **SMOTE (Synthetic Minority Over-sampling Technique)**：在少数类样本之间的连线上插值生成新样本，而非简单复制。是目前最常用的过采样方法。
- **代价敏感学习**：不对数据重采样，而是在损失函数中为少数类赋予更高的误分类代价（如 $cost_{FN} \gg cost_{FP}$）。
- **混合方法**：将过采样和欠采样结合使用，如 SMOTE + Tomek Links 或 SMOTE + ENN，先过采样少数类再清洗边界样本。

## 数学推导

**SMOTE 算法步骤**：
对于少数类中的每个样本 $x_i$：

- 找到 $x_i$ 的 $k$ 个最近邻（在少数类内部）
- 随机选择一个近邻 $\hat{x}_i$
- 在 $x_i$ 和 $\hat{x}_i$ 之间的连线上随机生成新样本：
$$
x_{\text{new}} = x_i + \lambda \cdot (\hat{x}_i - x_i), \quad \lambda \sim U[0, 1]
$$

重复直到少数类达到目标数量。

**SMOTE 的变体**：
- **Borderline-SMOTE**：只对决策边界附近的少数类样本做 SMOTE（更具信息量）
- **ADASYN**：自适应地生成样本——对更难学习的少数类样本生成更多合成样本

**代价敏感学习的损失函数**：
对于二分类，当少数类为正类时：
$$
Loss = \sum_{i \in Pos} w_{pos} \cdot L(y_i, \hat{y}_i) + \sum_{i \in Neg} w_{neg} \cdot L(y_i, \hat{y}_i)
$$
通常设置 $w_{pos} / w_{neg} \approx N_{neg} / N_{pos}$。

**随机欠采样的信息损失**：假设多数类有 $N_{maj}$ 个样本，少数类有 $N_{min}$ 个，欠采样后的多数类保留 $N_{min}$ 个。丢失的信息量为 $(N_{maj} - N_{min}) / N_{maj}$ 比例的样本，可能包含对决策边界重要的样本。

## 直观理解

- **SMOTE 的"合成"逻辑**：SMOTE 不像简单复制那样"复印"少数类样本（导致过拟合），而是在两个真实少数类样本之间"孕育"新样本——想象一棵树上两个相邻的果子，它们之间的空间可以自然生长出新的"中间态"果子。这迫使模型在少数类的分布空间内学到更平滑的决策边界。
- **欠采样 vs 过采样**：欠采样像"裁员"——把多数类的员工裁掉大部分，虽然资源省了但可能裁掉了有经验的老员工。过采样像"扩招"——给少数类增加更多新员工，但如果不合理扩招（简单复制），新员工"千人一面"没意义。SMOTE 像有质量的"培训"——让少数类员工之间相互学习，产生多样化的新人才。
- **混合策略的原因**：SMOTE 可能产生位于多数类区域内的噪声样本，Tomek Links 或 ENN 可以清理这些边界区域，就像"扩招后再筛选"——先扩充队伍，再清除那些"站错队"的人员。

## 代码示例

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# 生成不平衡数据（正类占 5%）
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05],
                           n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f"训练集分布: {np.bincount(y_train)}")

# 基线模型
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)
print(f"基线 Recall: {recall_score(y_test, y_pred):.4f}")
print(f"基线 F1: {f1_score(y_test, y_pred):.4f}")

# SMOTE 过采样
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nSMOTE 后分布: {np.bincount(y_train_sm)}")
model_sm = LogisticRegression().fit(X_train_sm, y_train_sm)
print(f"SMOTE Recall: {recall_score(y_test, model_sm.predict(X_test)):.4f}")
print(f"SMOTE F1: {f1_score(y_test, model_sm.predict(X_test)):.4f}")

# 欠采样
under = RandomUnderSampler(random_state=42)
X_train_un, y_train_un = under.fit_resample(X_train, y_train)
print(f"\n欠采样后分布: {np.bincount(y_train_un)}")

# SMOTE + ENN 混合
smote_enn = SMOTEENN(random_state=42)
X_train_comb, y_train_comb = smote_enn.fit_resample(X_train, y_train)
print(f"\nSMOTE+ENN 后分布: {np.bincount(y_train_comb)}")
```

## 深度学习关联

- **Focal Loss**：Focal Loss 是目标检测中处理正负类极度不平衡（负样本远多于正样本）的标准损失函数。它在交叉熵基础上引入调制因子 $(1-p_t)^\gamma$，降低易分类样本的权重，让模型聚焦于困难样本——与 SMOTE 的"关注少数类"思想一致，但通过调整损失而非重采样实现。
- **数据增强解决不平衡**：在深度学习中，过采样多通过数据增强 (Data Augmentation) 实现——对少数类样本做旋转、裁剪、颜色变换等操作生成多样化的新样本，比 SMOTE 更适合图像等高维数据。
- **GAN 生成少数类样本**：生成对抗网络 (GAN) 可用于生成高质量的少数类合成样本。例如在医学影像中，用 GAN 生成罕见的病变图像，在不平衡数据上训练分类器，效果优于 SMOTE。
