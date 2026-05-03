# 13_LightGBM 的直方图算法与 Leaf-wise 生长

## 核心概念

- **LightGBM**：微软推出的梯度提升框架，在 XGBoost 基础上进一步优化训练速度和内存占用，特别适合大规模数据。
- **直方图算法 (Histogram-based)**：将连续特征离散化为 $k$ 个桶（bin），用桶的统计值代替精确值来寻找最佳分裂点，将时间复杂度从 $O(\#data \times \#features)$ 降至 $O(\#bins \times \#features)$。
- **Leaf-wise 生长策略**：每次选择**增益最大的叶节点**进行分裂（而非 Level-wise 的逐层生长），在相同分裂次数下误差更低，但需要注意控制深度防止过拟合。
- **GOSS (Gradient-based One-Side Sampling)**：保留梯度大的样本（训练不足），随机采样梯度小的样本（已训练好的），在保持精度的同时减少数据量。
- **EFB (Exclusive Feature Bundling)**：将互斥的特征（几乎不同时取非零值的特征）捆绑为一个特征，进一步减少特征维度。
- **类别特征原生支持**：无需 One-Hot 编码，直接在类别上进行最优分裂搜索（如按梯度统计排序后二分）。

## 数学推导

**直方图分裂增益计算**：
将特征 $f$ 的取值范围划分为 $k$ 个桶 $B_1, \dots, B_k$。每个桶 $B_j$ 的梯度和 Hessian 和为：
$$
G_j = \sum_{i \in B_j} g_i, \quad H_j = \sum_{i \in B_j} h_i
$$

对于某个分裂点（在第 $s$ 个桶处划分），左侧和右侧的统计量为：
$$
G_L = \sum_{j \leq s} G_j, \quad H_L = \sum_{j \leq s} H_j, \quad G_R = \sum_{j > s} G_j, \quad H_R = \sum_{j > s} H_j
$$

分裂增益为：
$$
Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{(H_L + H_R) + \lambda} \right] - \gamma
$$

**GOSS 采样方法**：
设梯度绝对值最大的 $a \times 100\%$ 的样本为 $A$ 集合，从剩余样本中随机采样 $b \times 100\%$ 为 $B$ 集合。对 $B$ 集合的梯度乘以权重 $(1-a)/b$ 以保持分布无偏。信息增益估计为：
$$
\tilde{V}_j(d) = \frac{1}{n} \left( \frac{(\sum_{x_i \in A_L} g_i + \frac{1-a}{b} \sum_{x_i \in B_L} g_i)^2}{n_L} + \frac{(\sum_{x_i \in A_R} g_i + \frac{1-a}{b} \sum_{x_i \in B_R} g_i)^2}{n_R} \right)
$$

**EFB 特征捆绑**：
将互斥特征 $f_i, f_j$ 合并为新特征 $f_{ij}$，合并时对不同特征的值加上偏移量（如将 $f_j$ 的值加上 $f_i$ 的最大值），使合并后的值域不重叠。

## 直观理解

- **直方图 = 粗粒度搜索**：精确算法像用游标卡尺测量（精确但慢），直方图算法像用厘米尺（略粗糙但快得多）。实践证明，64-256 个桶足以找到接近最优的分裂点，速度提升数倍甚至数十倍。
- **Leaf-wise vs Level-wise**：Level-wise 逐层生长像修剪整齐的"平头"——同一层的所有节点同时分裂，不管某个节点是否需要；Leaf-wise 像"自由生长"——只分裂最有潜力的节点，同样分裂次数下精度更高，但可能长出不均衡的深树，需要限制 `max_depth`。
- **GOSS 的直觉**：梯度大的样本是"问题学生"，需要重点关注；梯度小的样本是"好学生"，可以抽样关注。把精力放在难学的样本上，效率自然更高。

## 代码示例

```python
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 加载数据
data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# LightGBM 回归
model = lgb.LGBMRegressor(n_estimators=100, max_depth=6,
                          learning_rate=0.1, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# 预测与评估
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"LightGBM RMSE: {rmse:.4f}")

# 特征重要性
print(f"特征重要性: {model.feature_importances_}")

# 直方图信息（特征被分桶的情况）
print(f"模型使用的特征数: {model.n_features_}")
```

## 深度学习关联

- **直方图思想在深度学习中**：Deep Learning 中的模型量化 (Quantization) 与 LightGBM 的直方图思想类似——将连续的权重和激活值离散化为低位表示（如 INT8），在保持精度的同时大幅提升推理速度。
- **GOSS 与难例挖掘**：GOSS 的思路——优先关注梯度大的"难样本"——与深度学习中的难例挖掘 (Hard Example Mining) 和 OHEM 算法一致。在目标检测、人脸识别等任务中，难例挖掘是提升模型精度的关键技术。
- **特征绑定的嵌入层启发**：EFB 将互斥特征捆绑的思路与深度推荐系统中的 Feature Embedding 类似——将高维稀疏特征映射到低维稠密空间，降低计算和存储成本。DCN、DeepFM 等模型也采用类似的特征交互优化策略。
