# 集成学习

## 一、基本思想

将多个弱学习器组合成一个强学习器，降低方差（Bagging）或偏差（Boosting）。

---

## 二、Bagging

并行训练多个模型，投票/平均。

### 2.1 随机森林

- 多个决策树，每棵树使用随机子集的样本和特征
- 降低方差，不易过拟合
- 支持特征重要性评估

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
```

---

## 三、Boosting

串行训练，每轮关注前一轮的错误。

### 3.1 AdaBoost

调整样本权重：错误分类的样本权重增加。

### 3.2 Gradient Boosting

每棵树拟合前一轮残差的负梯度方向。

### 3.3 XGBoost

GBDT的工程优化版：
- 二阶泰勒展开
- 正则化项
- 列采样
- 缺失值处理
- 并行化

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)
```

### 3.4 LightGBM

- 基于直方图的分裂
- GOSS（单侧梯度采样）
- EFB（互斥特征捆绑）
- 训练速度比XGBoost快数倍

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

### 3.5 CatBoost

- 自动处理类别特征
- Ordered Boosting减少过拟合
- 调参需求少

---

## 四、Stacking

用第一层多个模型的输出作为第二层（元学习器）的输入。

```
第一层：RF, SVM, KNN, XGBoost
    ↓ 输出概率
第二层：Logistic Regression / 神经网络
```
