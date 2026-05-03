# 21_准确率、精确率、召回率与 F1 Score

## 核心概念

- **准确率 (Accuracy)**：所有预测中正确的比例，$Accuracy = (TP + TN) / (TP + TN + FP + FN)$。在类别平衡时适用，但对不平衡数据有误导性。
- **精确率 (Precision)**：被预测为正类的样本中有多少是真正的正类，$Precision = TP / (TP + FP)$。衡量"查准率"——即模型说"是正类"时的可信度。
- **召回率 (Recall)**：真正的正类中有多少被正确识别，$Recall = TP / (TP + FN)$。衡量"查全率"——即模型找出所有正类的能力。
- **F1 Score**：精确率和召回率的调和平均数，$F1 = 2 \times (P \times R) / (P + R)$。同时兼顾精确率和召回率，是不平衡分类的常用综合指标。
- **精确率-召回率权衡**：提高精确率通常降低召回率，反之亦然（取决于分类阈值）。F1 在两者之间寻求平衡。
- **宏平均 vs 微平均**：多分类任务中，宏平均 (Macro-Avg) 计算每个类别的指标再平均（平等对待每个类），微平均 (Micro-Avg) 聚合所有类别的预测再计算（平等对待每个样本）。

## 数学推导

**二分类混淆矩阵**：
$$
\begin{matrix}
& \text{预测正} & \text{预测负} \\
\text{真实正} & TP & FN \\
\text{真实负} & FP & TN
\end{matrix}
$$

各指标定义：
- $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- $Precision = \frac{TP}{TP + FP}$（所有预测为正类中正确的比例）
- $Recall = \frac{TP}{TP + FN}$（所有真实正类中被找出的比例）
- $Specificity = \frac{TN}{TN + FP}$（所有真实负类中被正确识别的比例）

**F$\beta$ Score**（F1 的推广）：
$$
F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}
$$
$\beta = 1$ 时即为 F1，$\beta > 1$ 更看重 Recall，$\beta < 1$ 更看重 Precision。

**多分类宏平均 F1**：
$$
Macro\text{-}F1 = \frac{1}{C} \sum_{c=1}^C F1_c
$$

**多分类微平均 F1**：
计算全局的 $TP_{total}, FP_{total}, FN_{total}$，再求 F1：
$$
Micro\text{-}F1 = 2 \times \frac{P_{micro} \cdot R_{micro}}{P_{micro} + R_{micro}}
$$

其中 $P_{micro} = \frac{\sum TP_c}{\sum (TP_c + FP_c)}$，$R_{micro} = \frac{\sum TP_c}{\sum (TP_c + FN_c)}$。

**准确率的陷阱**：如果 95% 的样本是负类，全部预测为负类也能获得 95% 的准确率，但模型毫无价值。这正是精确率、召回率和 F1 分数存在的意义。

## 直观理解

- **精确率 vs 召回率的情景**：精确率关心"没有误伤"——垃圾邮件过滤宁愿漏掉一些垃圾邮件（低召回），也不要把正常邮件误判为垃圾（高精确）。召回率关心"没有漏网"——癌症筛查宁可误报（低精确），也不能漏掉任何一个病人（高召回）。
- **F1 的调和平均特性**：调和平均对较小值更敏感。如果 $P=1.0, R=0.01$，算术平均约 0.5，但 F1 只有约 0.02——正确反映了"模型几乎找不出正类"的糟糕表现。
- **宏平均 vs 微平均**：宏平均给每个类别同等投票权（小类也有话语权），微平均受大类主导。在类别极不平衡时，微平均 F1 可能虚高（主要反映大类表现），而宏平均 F1 更能揭示模型对少数类的表现。

## 代码示例

```python
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, fbeta_score)

# 模拟二分类结果
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(f"精确率: {precision_score(y_true, y_pred):.4f}")
print(f"召回率: {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
print(f"F2 Score (更看重Recall): {fbeta_score(y_true, y_pred, beta=2):.4f}")
print(f"F0.5 Score (更看重Precision): {fbeta_score(y_true, y_pred, beta=0.5):.4f}")

# 多分类指标
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred_multi = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 0])

print("\n多分类报告:")
print(classification_report(y_true_multi, y_pred_multi,
                            target_names=['类0', '类1', '类2']))
```

## 深度学习关联

- **分类任务的评估标准**：深度分类模型（图像分类、文本分类）的评估完全依赖这套指标。在多标签分类中，还需使用 micro/macro F1、Hamming Loss 等扩展指标。
- **目标检测中的 mAP**：目标检测任务的 mAP (mean Average Precision) 本质上是多个 Recall 阈值下的平均 Precision，是精确率-召回率曲线下面积的深度推广。
- **类别不平衡处理**：深度学习中处理不平衡数据时，Focal Loss 通过降低易分类样本的权重来提升困难样本的 Recall，与 F$\beta$ Score 中调整 $\beta$ 的思路一脉相承。
