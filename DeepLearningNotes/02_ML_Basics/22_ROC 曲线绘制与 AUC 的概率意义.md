# ROC 曲线绘制与 AUC 的概率意义

## 核心概念
- **ROC 曲线**：Receiver Operating Characteristic Curve，以**假正率 (FPR)** 为横轴、**真正率 (TPR, 即 Recall)** 为纵轴，绘制不同分类阈值下的性能表现。
- **真正率 (TPR)**：$TPR = TP / (TP + FN) = Recall$，正确识别正类的比例。
- **假正率 (FPR)**：$FPR = FP / (FP + TN)$，将负类错误识别为正类的比例。
- **AUC (Area Under the Curve)**：ROC 曲线下的面积，取值范围 [0.5, 1.0]，衡量模型的整体排序性能。
- **AUC 的概率解释**：AUC 等于随机从正类选一个样本、从负类选一个样本，模型将正类排到负类前面的概率。即 $AUC = P(\text{score}(x_+) > \text{score}(x_-))$。
- **阈值无关性**：ROC 曲线和 AUC 是**阈值无关**的评估指标，反映模型本身的排序能力，不依赖于特定的分类阈值。

## 数学推导
**TPR 和 FPR 的定义**：
$$
TPR = \frac{TP}{TP + FN} = \frac{\text{正确预测的正类数}}{\text{所有真实正类数}}
$$
$$
FPR = \frac{FP}{FP + TN} = \frac{\text{错误预测为正类的负类数}}{\text{所有真实负类数}}
$$

**AUC 的两种计算方法**：

1. **积分法（从曲线计算）**：
$$
AUC = \int_{0}^{1} TPR(FPR^{-1}(x)) dx
$$

2. **Wilcoxon-Mann-Whitney 统计量（概率解释）**：
$$
AUC = \frac{1}{m_+ m_-} \sum_{i=1}^{m_+} \sum_{j=1}^{m_-} \mathbb{I}(\text{score}(x_i^+) > \text{score}(x_j^-))
$$
其中 $m_+$ 为正类样本数，$m_-$ 为负类样本数。

**AUC 与 Gini 系数的关系**：
$$
Gini = 2 \times AUC - 1
$$

**ROC 曲线绘制步骤**：
1. 模型输出每个样本的预测概率（或分数）
2. 按分数从高到低排序
3. 从最高分到最低分依次作为阈值，计算每个阈值下的 TPR 和 FPR
4. 连接各点得到 ROC 曲线

**AUC 的方差**：AUC 的方差可通过 $SE(AUC) = \sqrt{\frac{AUC(1-AUC) + (m_+-1)(Q_1 - AUC^2) + (m_--1)(Q_2 - AUC^2)}{m_+ m_-}}$ 估计，其中 $Q_1 = AUC/(2-AUC)$，$Q_2 = 2AUC^2/(1+AUC)$。

## 直观理解
- **"随机排序测试"**：想象一个不懂分类但会排序的模型，给每个样本打一个随机分数。正类分数高于负类的概率是 50%，即 AUC=0.5（对角线）。完美模型的正类分数总是高于所有负类，AUC=1.0。
- **"是否停在角落"**：ROC 曲线越靠近左上角 (0,1)，模型越好。左上角意味着 TPR=1（所有正类都找出来了）且 FPR=0（没有一个误判），这是理想点。
- **阈值如何影响 ROC**：当阈值从 0 到 1 变化时，分类器从"所有样本都预测为正"（右上角 TPR=1, FPR=1）逐渐变为"所有样本都预测为负"（左下角 TPR=0, FPR=0）。ROC 曲线记录了中间的轨迹。

## 代码示例
```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, n_features=5,
                           n_informative=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

# 计算 ROC 和 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

# 概率解释验证：随机采样一对正负样本看排序正确率
pos_scores = y_scores[y_test == 1]
neg_scores = y_scores[y_test == 0]
correct_ordering = sum(p > n for p in pos_scores for n in neg_scores)
total_pairs = len(pos_scores) * len(neg_scores)
print(f"验证概率解释: {correct_ordering / total_pairs:.4f}")

# 阈值影响
idx = [0, len(thresholds)//4, len(thresholds)//2, 3*len(thresholds)//4, -1]
for i in idx:
    print(f"阈值={thresholds[i]:.3f}: TPR={tpr[i]:.3f}, FPR={fpr[i]:.3f}")
```

## 深度学习关联
- **排序损失 (Ranking Loss)**：深度推荐系统中的 AUC 优化常通过 pairwise 排序损失（如 Bayesian Personalized Ranking, BPR Loss）实现，直接优化 AUC 的概率解释——让正负样本对的排序正确。
- **AUC 作为早停指标**：在深度分类任务中，AUC 常作为早停 (Early Stopping) 的监测指标，因为 AUC 对阈值不敏感，能更稳定地反映模型学习进度。
- **AUC 在对抗性评估中**：在深度学习的鲁棒性评估中，AUC 被广泛用于衡量对抗样本攻击下模型的排序性能保持程度，作为分类准确率之外的重要鲁棒性指标。
