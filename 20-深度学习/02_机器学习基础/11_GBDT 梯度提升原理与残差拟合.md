# 11_GBDT 梯度提升原理与残差拟合

## 核心概念

- **梯度提升决策树 (GBDT)**：一种迭代式集成方法，每一轮训练一棵新的决策树来拟合当前模型的**负梯度**（即残差的方向），将新树加入模型以逐步减小损失。
- **加法模型**：GBDT 的预测结果是所有树的加权和 $F_M(x) = \sum_{m=1}^M T_m(x)$，每一棵新树都在修正前一阶段模型的不足。
- **残差拟合**：对于回归任务，当前模型的"残差" $y - F_{m-1}(x)$ 就是负梯度。新树的目标就是预测这些残差，相当于在弥补前序模型的错误。
- **梯度提升视角**：将问题泛化到任意可微损失函数，每轮用树去拟合损失函数在当前模型上的负梯度 $-\partial L / \partial F$ ——这正是"梯度提升"名称的来源。
- **学习率 (Shrinkage)**：每棵树乘以一个学习率 $\nu \in (0,1]$ 再累加，给后续树留下更多修正空间，是防止过拟合的关键手段。
- **对比随机森林**：RF 的树是**并行独立**训练的（Bagging），GBDT 的树是**串行依赖**训练的（Boosting）。RF 主要降低方差，GBDT 主要降低偏差。

## 数学推导

给定训练集 $\{(x_i, y_i)\}_{i=1}^m$ 和可微损失函数 $L(y, F(x))$。

**Step 1**: 初始化常数模型 $F_0(x) = \arg\min_\gamma \sum_{i=1}^m L(y_i, \gamma)$

对于 $m = 1$ 到 $M$：

**Step 2**: 计算负梯度（伪残差）：
$$
r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}}
$$

对于回归用 MSE 损失 $L(y, F) = \frac{1}{2}(y-F)^2$，则 $r_{im} = y_i - F_{m-1}(x_i)$，即残差。

**Step 3**: 用回归树拟合 $\{(x_i, r_{im})\}_{i=1}^m$，得到叶节点区域 $R_{jm}, j=1,\dots,J_m$。

**Step 4**: 对每个叶节点 $j$，计算最优输出值：
$$
\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)
$$

**Step 5**: 更新模型 $F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(x \in R_{jm})$

最终模型为 $F_M(x) = \sum_{m=0}^M \nu \cdot T_m(x)$。

## 直观理解

- **"亡羊补牢"的哲学**：GBDT 的做法很像考试后订正错题——第一轮考试得了 60 分，分析错题（残差）后重点补习薄弱知识点，第二轮考到 80 分，再分析新的错题……如此迭代，成绩越来越好。
- **梯度下降在函数空间**：梯度下降是在**参数空间**中沿着负梯度方向更新参数；GBDT 是在**函数空间**中沿着负梯度方向更新函数。每棵树相当于函数空间中的一个更新步。
- **为什么拟合残差有效？**：如果上一轮预测值是 100，真实值是 110，残差是 10。新树如果学会预测残差 10，叠加后预测值就变成 110，误差归零。虽然实践中残差会不断变化（后续树可能改变之前树的"分工"），但这个逐步逼近的核心思路是有效的。

## 代码示例

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

# 回归任务
X, y = make_regression(n_samples=200, n_features=5, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

gbdt_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
gbdt_reg.fit(X_train, y_train)
print(f"GBDT 回归 R^2: {gbdt_reg.score(X_test, y_test):.4f}")
print(f"训练迭代中的损失: 初始={gbdt_reg.train_score_[0]:.2f}, "
      f"最终={gbdt_reg.train_score_[-1]:.2f}")

# 分类任务
Xc, yc = make_classification(n_samples=500, n_features=8, random_state=42)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3)

gbdt_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
gbdt_clf.fit(Xc_tr, yc_tr)
print(f"GBDT 分类准确率: {gbdt_clf.score(Xc_te, yc_te):.4f}")
print(f"特征重要性: {gbdt_clf.feature_importances_}")
```

## 深度学习关联

- **ResNet 与残差连接**：深度残差网络 (ResNet) 的核心创新 $x_{l+1} = x_l + F(x_l)$ 与 GBDT 的加法模型 $F_m = F_{m-1} + T_m$ 在思想上完全一致——都是通过让后续模块拟合"残差"来构建更深的模型。可以说 ResNet 是 GBDT 在深度神经网络中的对应物。
- **DenseNet 与特征复用**：DenseNet 的密集连接——每层接收前面所有层的输出作为输入——类似于 GBDT 中每棵新树看到的是所有之前树累积的信息，体现了逐步精炼的增强思想。
- **LambdaRank / LambdaMART**：GBDT 在排序学习 (Learning to Rank) 中的经典应用 LambdaMART 直接启发了深度学习排序模型的设计。深度排序模型（如 DNN-based rankers）中的 pairwise/listwise 损失函数很大程度上源于 GBDT 排序框架。
