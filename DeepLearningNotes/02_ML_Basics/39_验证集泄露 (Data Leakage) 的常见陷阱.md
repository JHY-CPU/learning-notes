# 验证集泄露 (Data Leakage) 的常见陷阱

## 核心概念
- **验证集泄露 (Data Leakage)**：训练过程中模型间接接触到验证集或测试集的信息，导致性能评估过于乐观，在实际部署中表现远低于预期。
- **目标泄露 (Target Leakage)**：将预测时不可获得的未来信息或与目标直接相关的信息作为特征。例如，用"是否住院"预测"是否患病"——住院本身就是患病的后果。
- **数据泄露 (Data Leakage)**：训练集和验证集之间共享了不该共享的信息，如预处理时用全数据的统计量（标准化、PCA）而非仅用训练集。
- **时间泄露 (Temporal Leakage)**：在时间序列预测中，用未来的数据来预测过去——如用第 t+1 天的数据预测第 t 天的值。
- **重复样本泄露**：数据集中存在重复或高度相似的样本同时出现在训练集和验证集中（常见于数据增强或爬虫数据）。
- **泄露后果**：模型评估指标虚高，部署时实际性能远低于期望，导致"训练时表现完美、上线时惨不忍睹"。

## 典型陷阱案例

**陷阱 1：预处理时使用全局统计量**
```python
# 错误做法
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用全数据拟合
X_train, X_test = train_test_split(X_scaled)
# 正确做法
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**陷阱 2：特征选择时使用全数据**
```python
# 错误：用全数据做特征选择，然后再划分
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
X_train, X_test = train_test_split(X_selected, y)
# 正确：划分后再做特征选择
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector.fit(X_train, y_train)  # 只在训练集上做
```

**陷阱 3：时间序列中的未来信息**
```python
# 错误：随机划分时间序列数据
X_train, X_test = train_test_split(X, shuffle=True)
# 正确：按时间顺序划分
train_idx = df.index[df['date'] < '2022-01-01']
test_idx = df.index[df['date'] >= '2022-01-01']
```

**陷阱 4：数据增强泄露**
在图像分类中，对一张图片做旋转、裁剪等增强后，图片的增强版本可能同时出现在训练集和验证集中（如果增强是在划分前做的）。应先在原始数据上划分，再分别对训练集做数据增强。

**陷阱 5：重复数据**（去重是关键步骤）
```python
# 检查重复
print(f"重复的样本数: {data.duplicated().sum()}")
data = data.drop_duplicates()
```

**陷阱 6：ID 类特征**
`用户ID`、`交易ID` 等特征与目标变量高度相关（如 ID 为偶数的用户更容易违约，只是巧合），模型可能记住 ID 和目标之间的虚假关联。

## 数学解释

**泄露对评估指标的扭曲**：
假设在泄露的情况下，训练集和验证集的相关性从 $\rho=0$ 变为 $\rho=0.3$。模型在验证集上的表观性能：
$$
\text{Apparent AUC} = \Phi\left(\frac{\hat{\mu}_+ - \hat{\mu}_-}{\sqrt{\hat{\sigma}_+^2 + \hat{\sigma}_-^2}}\right)
$$
由于泄露，$\hat{\mu}_+ - \hat{\mu}_-$ 被高估（模型知道了不该知道的信息），导致 AUC 虚高。

**时间泄露对残差的影响**：
在时间序列中，如果用未来信息预测过去：
$$
y_t = \beta_0 + \beta_1 y_{t+1} + \varepsilon_t
$$
这个模型在"预测"时已经看到了 $y_{t+1}$，评估指标完美但毫无实际预测能力——因为真正的预测场景中无法获得未来数据。

## 代码示例
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 模拟泄露场景
np.random.seed(42)
n = 200
X = np.random.randn(n, 5)
y = (X[:, 0] + np.random.randn(n) * 0.5 > 0).astype(int)

# 泄露版本：标准化用全数据
scaler_leaked = StandardScaler()
X_scaled_leaked = scaler_leaked.fit_transform(X)
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X_scaled_leaked, y, test_size=0.3)

model_leaked = LogisticRegression()
model_leaked.fit(X_tr_l, y_tr_l)
acc_leaked = accuracy_score(y_te_l, model_leaked.predict(X_te_l))

# 正确版本
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
scaler_correct = StandardScaler()
X_tr_c = scaler_correct.fit_transform(X_tr)
X_te_c = scaler_correct.transform(X_te)

model_correct = LogisticRegression()
model_correct.fit(X_tr_c, y_tr_c)
acc_correct = accuracy_score(y_te_c, model_correct.predict(X_te_c))

print(f"泄露版本准确率: {acc_leaked:.4f} (虚高)")
print(f"正确版本准确率: {acc_correct:.4f} (真实)")

# 另一个泄露来源：使用未来信息
df = pd.DataFrame({
    'day': range(100),
    'value': np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
})
# 泄露：用明天的值预测今天
df['target'] = df['value'].shift(-1)  # 错误！
# 正确：用过去的值预测未来
df['target_correct'] = df['value'].shift(1)  # 正确
```

## 深度学习关联
- **数据增强中的泄露**：在深度学习数据增强中，必须确保增强后的图像不会同时出现在训练集和验证集中。例如，对一张图片做水平翻转和旋转后，增强版本和原始版本应全部在训练集内。常见的错误是增强后再划分数据。
- **验证集在早停中的作用**：深度学习训练中，验证集用于早停（Early Stopping）和模型选择。如果验证集信息通过任何方式"渗入"训练过程（如多次用验证集调整超参数），模型会间接过拟合验证集。解决方案是留出一个独立的测试集，只用于最终评估。
- **预训练模型中的泄露风险**：在 NLP 中，如果预训练模型（如 BERT）的训练语料包含了测试集数据，会导致评估不公平。这是大语言模型评估中的一个重要挑战——需要确保测试集是"模型未见过的"。
