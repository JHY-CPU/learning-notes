# 模型评估与选择

## 一、评估指标

### 1.1 分类指标

- **Accuracy**：正确预测比例，类别不平衡时不靠谱
- **Precision**：$\frac{TP}{TP+FP}$，预测为正中真正为正的比例
- **Recall**：$\frac{TP}{TP+FN}$，实际为正中被正确预测的比例
- **F1-Score**：Precision和Recall的调和平均
- **AUC-ROC**：ROC曲线下面积，衡量排序能力
- **PR曲线**：类别不平衡时比ROC更有信息量

### 1.2 回归指标

- **MSE**：均方误差
- **RMSE**：均方根误差
- **MAE**：平均绝对误差
- **MAPE**：平均绝对百分比误差
- **R² Score**：方差解释比例

---

## 二、交叉验证

- **K-Fold**：K=5或10常用
- **Stratified K-Fold**：保持类别比例
- **Leave-One-Out**：K=N，小数据集
- **时间序列**：不能随机划分，使用前向验证

---

## 三、超参数调优

### 3.1 网格搜索

穷举所有参数组合，计算量大。

### 3.2 随机搜索

随机采样参数组合，效率更高。

### 3.3 贝叶斯优化

用高斯过程建模目标函数，选择最有希望的参数。

```python
from optuna import create_study
study = create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## 四、模型选择

- 简单问题先试逻辑回归/决策树
- 结构化数据：XGBoost/LightGBM
- 图像：CNN/ViT
- 文本：Transformer/BERT
- 序列：RNN/LSTM/Transformer
- 没有免费午餐：需要实验验证
