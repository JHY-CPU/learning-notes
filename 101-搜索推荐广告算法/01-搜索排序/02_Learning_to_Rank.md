# Learning to Rank - 搜索推荐广告算法


## 1. LTR三类方法总览


Learning to Rank（LTR，学习排序）将排序问题转化为机器学习问题，根据训练数据学习最优的文档排序函数。


| 方法类别 | 建模粒度 | 输入 | 损失函数 | 代表算法 |
| --- | --- | --- | --- | --- |
| Pointwise | 单个文档 | 查询-文档对 | 回归/分类损失 | 线性回归、LR、RankProp |
| Pairwise | 文档对 | 一对相关性不同的文档 | 偏好顺序损失 | RankNet、LambdaRank、LambdaMART |
| Listwise | 整个列表 | 查询对应的文档列表 | 列表级排序指标 | ListNet、SoftRank、AdaRank |


> **Note:** 从Pointwise到Listwise，方法越来越直接优化排序指标，但训练复杂度也相应增加。工业界最常用的是Pairwise方法（尤其是LambdaMART）。


## 2. Pointwise方法


Pointwise将排序问题转化为回归或分类问题，独立地对每个查询-文档对打分。


### 基本思路


- 输入：(query, document)对的特征向量
- 输出：相关性分数或相关性等级
- 训练：最小化预测分数与真实相关性之间的损失
- 推理：对每个文档独立打分，按分数排序


> **Example:** **Pointwise的局限：**
> 忽略了文档之间的相对关系。例如，同一查询下的两个文档，一个相关一个不相关，Pointwise只关注各自的绝对打分，不关注它们的排序关系。


## 3. Pairwise方法：RankNet与LambdaRank


### RankNet（Microsoft, 2005）


RankNet将排序问题转化为文档对的偏好预测问题。


$$
P(D_i ≻ D_j) = σ(s_i - s_j) = 1 / (1 + exp(-(s_i - s_j)))
                损失函数：L = -P̄_ij × log(P_ij) - (1 - P̄_ij) × log(1 - P_ij)
                其中 P̄_ij 为文档对的真实偏好概率
$$


### LambdaRank（Microsoft, 2006）


LambdaRank在RankNet基础上，将梯度与排序指标（如NDCG）的变化直接关联。


$$
λ_ij = ∂L/∂s_i = -σ(1 - P_ij) × |ΔNDCG|
                其中 |ΔNDCG| 是交换文档i和j后NDCG的变化量的绝对值
$$


> **Important:** **LambdaRank的核心创新：**
> 不直接推导损失函数，而是设计梯度（Lambda）。每个文档对的梯度乘以|ΔNDCG|，使得对排序指标影响大的文档对获得更大的梯度，直接优化排序质量。


## 4. LambdaMART：GBDT + Lambda梯度


LambdaMART是LambdaRank与GBDT（梯度提升决策树）的结合，是工业界最广泛使用的LTR算法。


| 组件 | 作用 |
| --- | --- |
| Lambda梯度 | 计算每个样本的梯度，考虑排序指标变化 |
| GBDT | 使用Lambda梯度训练决策树，拟合排序函数 |
| 多棵树集成 | 多棵决策树的预测值累加为最终排序分数 |


### LightGBM LambdaMART实现


```
import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

# 准备数据（每个查询下有多个文档）
# X: 特征矩阵 (n_samples, n_features)
# y: 相关性标签 (n_samples,)
# group: 每个查询的文档数量列表
np.random.seed(42)
n_queries = 100
groups = np.random.randint(5, 30, size=n_queries)  # 每个查询5-30个文档
n_samples = groups.sum()

X = np.random.randn(n_samples, 15)  # 15个特征
y = np.random.randint(0, 5, size=n_samples)  # 0-4级相关性

# 划分训练/测试集（按查询划分）
split_idx = int(n_queries * 0.8)
train_groups = groups[:split_idx]
test_groups = groups[split_idx:]
train_size = train_groups.sum()
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建LightGBM数据集
train_set = lgb.Dataset(X_train, y_train, group=train_groups)
test_set = lgb.Dataset(X_test, y_test, group=test_groups)

# LambdaMART参数
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10, 20],  # 评估NDCG@5, @10, @20
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'lambdarank_truncation_level': 10,  # Lambda计算中考虑的文档数量
    'verbose': -1,
}

# 训练
model = lgb.train(
    params,
    train_set,
    num_boost_round=500,
    valid_sets=[test_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# 预测与评估
y_pred = model.predict(X_test)

# 计算NDCG（需要按查询分组）
offset = 0
ndcg_scores = []
for g in test_groups:
    y_true_group = y_test[offset:offset+g].reshape(1, -1)
    y_pred_group = y_pred[offset:offset+g].reshape(1, -1)
    if y_true_group.sum() > 0:  # 至少有一个正样本
        ndcg_scores.append(ndcg_score(y_true_group, y_pred_group, k=10))
    offset += g

print(f"NDCG@10: {np.mean(ndcg_scores):.4f}")

# 特征重要性
importance = model.feature_importance(importance_type='gain')
for i, imp in enumerate(sorted(enumerate(importance), key=lambda x: -x[1])[:5]):
    print(f"Feature {imp[0]}: importance={imp[1]:.0f}")
```


## 5. Listwise方法


| 方法 | 核心思想 | 损失函数 |
| --- | --- | --- |
| ListNet | 将文档排列视为概率分布 | Cross entropy between probability distributions |
| SoftRank | 将排序位置视为随机变量 | 期望NDCG的梯度 |
| AdaRank | 直接优化NDCG等指标 | 基于指数损失的boosting |
| DLPR | 深度学习列表排序 | 基于注意力的列表级损失 |


> **Note:** Listwise方法直接以列表为单位计算损失，最直接地优化排序指标。但训练复杂度较高（需要考虑所有排列），且对大规模数据集不太友好。


## 6. LTR特征设计


| 特征类别 | 具体特征 | 说明 |
| --- | --- | --- |
| 查询-文档相关性 | BM25分数、TF-IDF分数、语义匹配分数 | 查询与文档的相关程度 |
| 文档质量 | PageRank、域名权威性、内容长度、更新时间 | 文档本身的质量指标 |
| 新鲜度 | 发布时间距今天数、更新频率 | 时效性需求的场景 |
| 用户行为 | 点击率、停留时间、历史点击 | 用户的隐式反馈信号 |
| 查询特征 | 查询长度、查询词频、意图分类 | 查询本身的特征 |


## 总结


- LTR将排序问题转化为机器学习问题，分为Pointwise、Pairwise、Listwise三类方法
- Pairwise方法（LambdaRank、LambdaMART）是工业界主流，直接优化排序指标
- LambdaMART结合了Lambda梯度和GBDT，具有可解释性强、效果好的优势
- LightGBM的lambdarank目标函数是LambdaMART的高效实现
- Listwise方法直接优化列表级指标，但训练复杂度较高
- 特征设计是LTR成功的关键，需要综合考虑相关性、质量、新鲜度、用户行为等维度


<!-- Converted from: 02_Learning_to_Rank.html -->
