# 12_XGBoost 目标函数泰勒展开与正则化

## 核心概念

- **XGBoost**：eXtreme Gradient Boosting，对 GBDT 的高效工程实现和算法改进，是 Kaggle 竞赛中的常胜将军。
- **二阶泰勒展开**：XGBoost 对损失函数进行二阶泰勒展开，同时利用梯度和 Hessian 矩阵（一阶和二阶信息）来指导树的构建，收敛更快。
- **显式正则化**：在目标函数中加入树的复杂度惩罚项 $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$，其中 $T$ 是叶节点数，$w_j$ 是叶节点权重。
- **列抽样 (Column Subsampling)**：借鉴随机森林的特征采样策略，每棵树或每次分裂时随机选择部分特征，降低过拟合。
- **Shrinkage 与缩减**：同 GBDT 的学习率，每棵树乘以权重系数再累加，为后续树留出学习空间。
- **工程优化**：包括特征级并行扫描（预排序分桶找最优分裂点，树之间仍是串行）、缓存感知访问、核外计算、分块压缩等，使 XGBoost 比原始 GBDT 快 10 倍以上。

## 数学推导

在第 $t$ 轮迭代，模型 $F_t(x) = F_{t-1}(x) + f_t(x)$，其中 $f_t$ 是第 $t$ 棵回归树。

目标函数为：
$$
Obj^{(t)} = \sum_{i=1}^m L(y_i, F_{t-1}(x_i) + f_t(x_i)) + \Omega(f_t)
$$

对损失函数做二阶泰勒展开 $L(y, F_{t-1} + f_t) \approx L(y, F_{t-1}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)$，其中：
$$
g_i = \frac{\partial L(y_i, F_{t-1}(x_i))}{\partial F_{t-1}(x_i)}, \quad h_i = \frac{\partial^2 L(y_i, F_{t-1}(x_i))}{\partial^2 F_{t-1}(x_i)}
$$

去掉常数项 $L(y_i, F_{t-1})$ 后，简化目标为：
$$
\tilde{Obj}^{(t)} = \sum_{i=1}^m \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2
$$

将属于叶节点 $j$ 的样本集定义为 $I_j = \{i : f_t(x_i) = w_j\}$，目标函数整理为：
$$
\tilde{Obj}^{(t)} = \sum_{j=1}^T \left[ \left( \sum_{i \in I_j} g_i \right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T
$$

对 $w_j$ 求最优解：
$$
w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

代入得最小目标值（结构分数）：
$$
Obj^* = -\frac{1}{2} \sum_{j=1}^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$

分裂时的增益计算：
$$
Gain = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
$$

只有当 $Gain > 0$ 时才会进行分裂，$\gamma$ 自然地起到了预剪枝的作用。

## 直观理解

- **二阶信息的优势**：一阶梯度只知道"该往哪走"，二阶 Hessian 还知道"该走多快"。好比下山时，只看坡度（一阶）知道要往下走；结合地面曲率（二阶）还能判断前方是不是悬崖，从而调整步幅。这使 XGBoost 收敛更快、更稳定。
- **正则化的意义**：$\gamma$ 控制叶节点数量，$\lambda$ 控制叶节点权重的大小。这就像一个园丁——$\gamma$ 限制树的分支数量（别长太多枝丫），$\lambda$ 限制每片叶子的大小（别让某片叶子长得太大），共同防止过拟合。
- **结构分数**：$Obj^*$ 衡量一棵树的质量（越低越好）。建树时 XGBoost 用这个分数来评估每个分裂点的质量，而不是像传统 GBDT 那样只靠 MSE 减少量。这使 XGBoost 能处理更广泛的损失函数。

## 代码示例

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# XGBoost 回归
model = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                         learning_rate=0.1, reg_lambda=1.0,
                         gamma=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"特征重要性:\n{model.feature_importances_}")

# 查看损失函数中的一阶和二阶梯度(手动计算 MSE 的梯度和 Hessian)
def mse_gradient_hessian(y_true, y_pred):
    grad = y_pred - y_true  # 一阶导数
    hess = np.ones_like(y_true)  # 二阶导数为常数 1
    return grad, hess
```

## 深度学习关联

- **二阶优化方法**：深度学习中使用的牛顿法、L-BFGS 等二阶优化方法与 XGBoost 的二阶泰勒展开共享相同的理论基础。在深度学习中，Adam 等自适应优化器也结合了梯度的一阶矩和二阶矩估计，与 XGBoost 的 $g_i, h_i$ 框架异曲同工。
- **正则化路径**：XGBoost 中 $\lambda$ 对叶节点权重的 $L_2$ 正则化，对应深度学习中的 Weight Decay。两者都是通过约束参数范数来防止过拟合，数学形式完全相同。
- **TabNet / NODE**：近年来出现的面向表格数据的深度学习模型（如 TabNet、NODE）在设计时大量借鉴了 XGBoost 的思想——特征选择、分裂决策的软性模拟、正则化策略等。可以说，XGBoost 仍是表格数据深度学习方法的重要基准和灵感来源。
