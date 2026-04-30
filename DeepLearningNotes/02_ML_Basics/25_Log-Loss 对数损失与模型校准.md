# Log-Loss 对数损失与模型校准

## 核心概念
- **对数损失 (Log-Loss)**：即交叉熵损失 (Cross-Entropy Loss)，衡量模型预测概率分布与真实标签分布之间的差异。值越小表示概率预测越准确。
- **数学定义**：二分类 $LogLoss = -\frac{1}{m} \sum [y \ln \hat{p} + (1-y) \ln(1-\hat{p})]$；多分类 $LogLoss = -\frac{1}{m} \sum \sum y_{ic} \ln \hat{p}_{ic}$。
- **模型校准 (Calibration)**：模型的预测概率是否与真实的频率一致——即预测概率为 0.8 的样本中，是否大约 80% 确实为正类。
- **校准曲线 (Calibration Curve)**：也称可靠性图 (Reliability Diagram)，将预测概率分箱，绘制每个箱中平均预测概率 vs 实际正类比例。
- **Brier Score**：另一种概率预测评估指标，$BS = \frac{1}{m} \sum (\hat{p}_i - y_i)^2$，范围 [0,1]，越小越好。
- **过自信 vs 欠自信**：深度模型普遍存在过自信 (Overconfidence) 问题——预测概率过于极端（接近 0 或 1），与实际频率不符。

## 数学推导
**二分类 Log-Loss**：
$$
LogLoss = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i)]
$$

当 $\hat{p}_i \to y_i$ 时，LogLoss → 0；当 $\hat{p}_i \to 1-y_i$ 时，LogLoss → ∞。

**多分类 Log-Loss**：
$$
LogLoss = -\frac{1}{m} \sum_{i=1}^m \sum_{c=1}^C y_{ic} \log(\hat{p}_{ic})
$$
其中 $y_{ic}$ 是 one-hot 标签，$\hat{p}_{ic}$ 是预测概率。

**Log-Loss 对概率的要求**：Log-Loss 对预测概率的"自信度"敏感——若模型预测 0.9 但猜错了，损失为 $-\log(0.1) \approx 2.3$；若预测 0.6 猜错，损失为 $-\log(0.4) \approx 0.9$。因此 Log-Loss 惩罚过自信的错误。

**预期校准误差 (ECE)**：
$$
ECE = \sum_{k=1}^K \frac{|B_k|}{m} |\text{acc}(B_k) - \text{conf}(B_k)|
$$
其中 $B_k$ 是第 $k$ 个概率箱，$\text{acc}(B_k)$ 是箱中样本的实际正类比例，$\text{conf}(B_k)$ 是平均预测概率。

**Platt Scaling (等温回归)** 校准方法：
$$
\hat{q}_i = \sigma(A \cdot \hat{p}_i + B)
$$
在验证集上学习参数 $A, B$（通过最小化 Log-Loss），将原始概率映射到校准后的概率。

## 直观理解
- **"诚实"的概率 vs "虚张声势"的概率**：Log-Loss 像一个严厉的考官——如果你说"80% 把握"却错了，扣分比说"60% 把握"错了要多得多。它奖励诚实、惩罚虚张声势。
- **校准的"天气预报"类比**：如果天气预报说"明天降雨概率 70%"，在 100 个这样的预报中，大约应该有 70 天真的下雨。如果实际只有 30 天下雨，那预报就是过自信的（高估概率）。完美的校准意味着预测概率与实际频率完全一致。
- **为什么 Log-Loss 重要？**：考虑两个分类器，准确率都是 90%，但 A 预测正确时自信（概率 0.99）、错误时迷茫（概率 0.51）；B 预测正确时也不太自信（概率 0.6）。A 的 Log-Loss 更低，因为它在正确时更有把握。在很多只需要排序的场景中，两者没区别（准确率相同），但在需要概率估计的场景中（如风控），A 显然更好。

## 代码示例
```python
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=500, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_probs = rf.predict_proba(X_test)[:, 1]

# Log-Loss
print(f"Log-Loss: {log_loss(y_test, y_probs):.4f}")

# Brier Score
print(f"Brier Score: {brier_score_loss(y_test, y_probs):.4f}")

# 校准曲线
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=5)
print(f"平均预测概率: {prob_pred}")
print(f"实际正类比例: {prob_true}")

# Platt 校准
calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
calibrated.fit(X_train, y_train)
cal_probs = calibrated.predict_proba(X_test)[:, 1]
print(f"校准后 Log-Loss: {log_loss(y_test, cal_probs):.4f}")
```

## 深度学习关联
- **神经网络过自信问题**：现代深度网络（尤其是使用 ReLU 和 BatchNorm 的网络）普遍存在过自信问题。Temperature Scaling 是深度学习中最常用的校准方法——在 Softmax 前除以温度参数 $T > 1$，使概率分布更平滑。
- **知识蒸馏中的温度**：Hinton 的知识蒸馏 (Knowledge Distillation) 使用高温 Softmax 生成软标签——$T$ 控制概率分布的"软硬程度"，高温下概率更均匀，保留了类别间的相似性信息（如"猫"比"汽车"更像"老虎"）。
- **不确定性估计**：在自动驾驶、医疗诊断等安全关键场景中，模型不仅需要给出预测，还需要给出可靠的不确定性估计。MC Dropout、Deep Ensemble 等方法通过多次前向传播估计预测的不确定性，与 Log-Loss 和校准评估密切相关。
