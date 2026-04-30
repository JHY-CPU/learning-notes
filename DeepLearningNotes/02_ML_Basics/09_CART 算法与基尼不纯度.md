# CART 算法与基尼不纯度

## 核心概念
- **CART 算法**：Classification and Regression Tree，由 Breiman 于 1984 年提出。核心特点是生成的树为**二叉树**（无论特征是离散还是连续），可同时用于分类和回归。
- **基尼不纯度 (Gini Impurity)**：衡量数据集纯度的指标，$Gini(D) = 1 - \sum_{k=1}^K p_k^2$。基尼系数越小，数据集纯度越高。
- **基尼增益**：特征 $A$ 分裂后的基尼不纯度加权和 $Gini(D, A) = \sum_{v=1}^V \frac{|D_v|}{|D|} Gini(D_v)$，选择使基尼增益最大的切分点。
- **二叉树二分**：CART 对每个特征做二分切分。对连续特征，在相邻值中点尝试切分；对多类别离散特征，按类别子集二分。
- **回归树**：对于回归任务，叶节点输出为落入该叶节点的样本均值，分裂准则为最小化均方误差 (MSE)。
- **代价复杂度剪枝 (CCP)**：CART 使用 CCP 进行后剪枝，用参数 $\alpha$ 权衡树的复杂度（叶节点数）与拟合误差。

## 数学推导
基尼不纯度的定义：
$$
Gini(D) = 1 - \sum_{k=1}^K p_k^2 = \sum_{k=1}^K p_k(1-p_k) = \sum_{k \neq j} p_k p_j
$$
其中 $p_k = |C_k| / |D|$ 是第 $k$ 类样本的比例。基尼系数可以理解为"从数据集中随机抽取两个样本，其类别不同的概率"。

对于特征 $A$ 的二分切分，分裂后的基尼不纯度：
$$
Gini(D, A) = \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2)
$$

**回归树**的分裂准则：对于节点 $R_m$ 包含 $N_m$ 个样本，预测值 $\hat{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$，损失为：
$$
L = \sum_{i \in R_m} (y_i - \hat{y}_m)^2
$$
分裂时最小化左右子节点的加权 MSE 和。

**CCP 剪枝**的损失函数：
$$
C_\alpha(T) = \sum_{t=1}^{|T|} \sum_{i \in R_t} (y_i - \hat{y}_t)^2 + \alpha |T|
$$
其中 $|T|$ 是叶节点数，$\alpha$ 是正则化参数。$\alpha$ 越大，树越简单。

**基尼系数 vs 信息熵**对比：在二分类中，$Gini = 2p(1-p)$，$Entropy = -p\log_2 p - (1-p)\log_2(1-p)$。两者形状相似，但基尼系数的计算不涉及对数，计算速度更快——这是 CART 选择基尼系数的实际原因之一。

## 直观理解
- **基尼系数的含义**：想象你在一个袋子里随机摸两个球。如果袋子里只有一种颜色的球，两次摸到不同颜色的概率为 0（纯度最高）；如果颜色均匀混合，摸到不同颜色的概率最大（纯度最低）。基尼系数就是"抽到不同类别"的概率。
- **二叉树的好处**：多路分裂可能导致数据被切得太碎，某些分支样本太少（数据碎片化）。CART 的二叉树策略每次只做一个二分决策，更加稳健，也更容易处理连续特征。
- **剪枝类似于"修剪树枝"**：一棵树长得太茂密（太深），虽然能完美拟合训练数据，但经不起风雨（过拟合）。CCP 剪枝就像园丁修剪掉那些"贡献不大"的枝叶，让树更简洁、更健康（泛化能力更强）。

## 代码示例
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_regression

# 分类：CART 使用基尼系数
iris = load_iris()
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
tree_clf.fit(iris.data, iris.target)
print(f"分类树特征重要性: {tree_clf.feature_importances_}")
print(f"训练准确率: {tree_clf.score(iris.data, iris.target):.3f}")

# 回归：CART 回归树
X_reg, y_reg = make_regression(n_samples=100, n_features=3, noise=0.1)
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X_reg, y_reg)
print(f"\n回归树 R^2 分数: {tree_reg.score(X_reg, y_reg):.4f}")

# 手动计算基尼系数
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)

# 纯节点 vs 不纯节点
print(f"\n纯节点基尼系数: {gini(np.array([0, 0, 0])):.4f}")
print(f"均匀混合基尼系数: {gini(np.array([0, 1, 2])):.4f}")
```

## 深度学习关联
- **梯度提升树的基函数**：XGBoost、LightGBM 等现代梯度提升框架使用 CART 回归树作为基学习器，每棵树拟合前一步的残差。这些树的二叉树结构和基尼分裂准则是整个集成方法的基础。
- **树与特征交互学习**：CART 树天然能捕捉特征之间的非线性交互（通过多层分裂），这一特点启发了深度学习中的特征交叉网络，如 DeepFM、xDeepFM 等模型通过显式构造特征交互层来模拟树的分裂效果。
- **可解释 AI (XAI)**：CART 树的决策路径完全透明，每条路径对应一个 if-then 规则集合。在需要高可解释性的深度系统中（如医疗、金融），常用 CART 树作为复杂深度模型的代理解释器 (Surrogate Explainer)。
