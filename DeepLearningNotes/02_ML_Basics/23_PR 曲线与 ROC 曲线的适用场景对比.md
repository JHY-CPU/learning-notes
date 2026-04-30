# PR 曲线与 ROC 曲线的适用场景对比

## 核心概念
- **PR 曲线**：Precision-Recall Curve，以召回率 (Recall) 为横轴、精确率 (Precision) 为纵轴，展示不同阈值下精确率和召回率的权衡关系。
- **ROC 曲线**：以假正率 (FPR) 为横轴、真正率 (TPR) 为纵轴的曲线。
- **核心区别**：PR 曲线关注正类的预测质量（精确率和召回率），ROC 曲线同时考虑正类和负类的综合表现。
- **不平衡数据敏感性**：当负类样本远多于正类时，ROC 曲线可能过于乐观（FPR 分母大，FP 的绝对增长对 FPR 影响小），而 PR 曲线直接反映正类的表现，对不平衡更敏感。
- **适用场景建议**：正负类平衡或关心整体排序 → ROC & AUC；正类稀缺或更关心正类预测质量（如异常检测、医疗诊断）→ PR 曲线。
- **曲线下面积**：PR-AUC（Average Precision）衡量 PR 曲线下面积，ROC-AUC 衡量 ROC 曲线下面积。

## 数学推导
**PR 曲线与 ROC 曲线的公式对比**：

PR 曲线涉及的两个指标只关注正类：
$$
Precision = \frac{TP}{TP + FP}, \quad Recall = \frac{TP}{TP + FN}
$$

ROC 曲线涉及的一个指标同时关注负类：
$$
TPR = \frac{TP}{TP + FN} = Recall, \quad FPR = \frac{FP}{FP + TN}
$$

**PR-AUC (Average Precision)** 的计算：
$$
AP = \sum_{n} (R_n - R_{n-1}) P_n
$$
其中 $R_n$ 和 $P_n$ 是第 $n$ 个阈值对应的召回率和精确率。这也等价于在不同 Recall 水平上对 Precision 插值后的积分。

**ROC 曲线不受不平衡影响的数学原因**：
当负类数量 $N$ 增加（不平衡加剧）：
- FPR 公式中的 $TN$ 和 $FP$ 都增加，但 $TN$ 是分母的主要部分，FPR 变化不大
- ROC 曲线基本保持不变（只要模型排序能力不变）
- 但 PR 曲线会显著下降，因为 $FP$ 绝对数量增加 → $Precision = TP/(TP+FP)$ 下降

**何时 PR 优于 ROC**：在极度不平衡数据中，如果 ROC-AUC 很高但 PR-AUC 很低，说明模型虽然能区分正负类（排序好），但对正类的预测精确度低（误报多）。这在某些场景下不可接受。

## 直观理解
- **"放大镜"效应的区别**：想象在草丛中找一枚硬币（正类）。ROC 曲线像站在远处看——大部分地面（负类）都是空的，误报（多挖几处）对 FPR 影响很小，看起来还不错。PR 曲线像蹲下来仔细找——每挖一处空地（FP）都浪费了时间，精确率明显下降，更能反映找硬币的实际难处。
- **什么时候用哪个？**：如果"把负类错判为正类"的代价小（如推荐系统推荐了不感兴趣的商品，用户只是忽略），ROC 曲线合适。如果"把负类错判为正类"的代价大（如将健康人诊断为癌症，会带来心理创伤和治疗成本），PR 曲线更合适。
- **两种曲线的阅读**：ROC 曲线靠近左上角好，PR 曲线靠近右上角好（Precision 和 Recall 都很高）。PR 曲线在正类稀少时通常看起来"差"很多，这反映了任务本身的难度。

## 代码示例
```python
import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成不平衡数据（正类占 5%）
X, y = make_classification(n_samples=1000, n_features=10,
                           weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_scores)
# PR-AUC (Average Precision)
pr_auc = average_precision_score(y_test, y_scores)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC (AP): {pr_auc:.4f}")
print(f"正类比例: {y_test.mean():.4f}")

# 对比：当正类比例变化时的 AUC 变化
for imbalance in [0.5, 0.2, 0.1, 0.05]:
    Xb, yb = make_classification(n_samples=1000, weights=[1-imbalance, imbalance])
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3)
    model.fit(Xb_tr, yb_tr)
    yb_scores = model.predict_proba(Xb_te)[:, 1]
    print(f"正类={imbalance:.0%}: ROC-AUC={roc_auc_score(yb_te, yb_scores):.3f}, "
          f"PR-AUC={average_precision_score(yb_te, yb_scores):.3f}")
```

## 深度学习关联
- **目标检测中的 PR 曲线**：目标检测领域的 mAP (mean Average Precision) 本质上是 PR-AUC 的改进版本——在不同 IoU 阈值下计算 PR-AUC 再平均。这体现了 PR 曲线在正类（目标框）稀少场景中的主导地位。
- **异常检测评估**：深度异常检测（如 Deep SVDD、GANomaly）的正类通常是极稀少异常样本，PR 曲线是最常用的评估指标，能真实反映模型在少数类上的表现。
- **医疗影像中的选择**：在医疗 AI 场景中，准确发现病灶（高 Recall）和减少误诊（高 Precision）同样重要。PR 曲线能同时反映这两个因素，因此是病理切片分析、CT 图像诊断等深度模型的主要评估工具。
