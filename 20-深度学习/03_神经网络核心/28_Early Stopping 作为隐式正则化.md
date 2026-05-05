# 29_Early Stopping 作为隐式正则化

## 核心概念

- **Early Stopping 的定义**：Early Stopping（早停法）是在训练过程中监控验证集性能，当验证指标不再改善时提前终止训练。它是最简单、最常用的防止过拟合技术之一。
- **隐式正则化的机制**：Early Stopping 限制了模型的有效训练步数，从而限制了模型的"有效复杂度"。在优化轨迹上，参数从初始值出发，越晚停止参数的"移动距离"越大，模型复杂度越高。Early Stopping 在参数空间中相当于引入了一个以初始值为中心的半径约束——这与 L2 正则化有深刻的联系。
- **与 L2 正则化的等价性**：对于使用梯度下降训练的线性模型，Early Stopping 在验证集上找到的解等价于 L2 正则化的解。停止时间 $T$ 与 L2 强度 $\lambda$ 之间存在一一对应关系：$\lambda \propto 1/T$。
- **验证集的关键作用**：Early Stopping 需要一个独立的验证集来监控泛化性能。验证集不能参与训练，否则会过拟合验证集。实践中通常从训练集中分出 10-20% 作为验证集。

## 数学推导

**线性模型的 Early Stopping 分析**：

考虑线性模型 $f(x) = \theta^T x$，使用梯度下降最小化 MSE：

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t), \quad L(\theta) = \frac{1}{2}\|X\theta - y\|^2
$$

从 $\theta_0 = 0$ 开始，梯度下降的迭代轨迹为：

$$
\theta_t = \theta_{t-1} - \eta X^T(X\theta_{t-1} - y)
$$

这个迭代的解可以写成：

$$
\theta_t = \sum_{i=1}^{t-1} (I - \eta X^T X)^i \eta X^T y
$$

假设 $X^T X$ 的特征分解为 $V\Lambda V^T$，则：

$$
\theta_t = V \cdot \frac{I - (I - \eta\Lambda)^t}{\Lambda} \cdot V^T X^T y
$$

每个特征模式 $k$ 的学习轨迹为：

$$
\theta_{t,k} = \frac{1 - (1 - \eta\lambda_k)^t}{\lambda_k} \cdot (V^T X^T y)_k
$$

其中 $\lambda_k$ 是第 $k$ 个特征值。

**与 L2 正则化的对比**：

L2 正则化解 $\theta_{L2}^* = (X^T X + \lambda I)^{-1} X^T y$

在特征模式下：

$$
\theta_{L2,k}^* = \frac{1}{\lambda_k + \lambda} \cdot (V^T X^T y)_k
$$

Early Stopping 解：

$$
\theta_{ES,k} = \frac{1 - (1 - \eta\lambda_k)^t}{\lambda_k} \cdot (V^T X^T y)_k
$$

两者对特征模式的衰减方式不同：
- L2：对所有模式统一加上 $\lambda$，大特征值模式衰减比例小
- Early Stopping：小特征值模式（对应噪声方向）尚未充分学习就被停止

当 $1 - (1 - \eta\lambda_k)^t \approx \frac{t\eta\lambda_k}{1 + t\eta\lambda_k}$ 时：

$$
\theta_{ES,k} \approx \frac{1}{\lambda_k + 1/(t\eta)} \cdot (V^T X^T y)_k
$$

这表明 $t$ 步 Early Stopping 等价于 L2 强度 $\lambda = 1/(t\eta)$。

## 直观理解

Early Stopping 的隐式正则化可以理解为"不要训练太久"——就像考试复习，复习太少了（欠拟合）不行，但复习太多了（过拟合）会将训练集的细节（噪声）也记住，导致考试时遇到新题目反而表现不好。

从优化轨迹的角度看，参数从初始值（通常靠近 0）出发，沿着梯度方向在参数空间中移动。早期移动方向对应数据的"主信号方向"（大特征值），后期移动方向逐渐包含了"噪声方向"（小特征值）。Early Stopping 在模型刚开始学习噪声之前就停止，因此实现了正则化效果。

Early Stopping 与 L2 正则化的等价性有一个直观解释：L2 正则化限制了参数可以偏离 0 的最大距离（通过惩罚力度 $\lambda$），而 Early Stopping 限制了参数从 0 出发的移动时间（通过步数 $t$）。在梯度下降中，两者达到的效果类似。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Early Stopping 实现
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# 演示 Early Stopping 的正则化效果
torch.manual_seed(42)

# 生成数据：一个简单的回归问题
n_train, n_val, n_test = 200, 100, 100
X = torch.randn(n_train + n_val + n_test, 20)
true_w = torch.cat([torch.ones(5), torch.zeros(15)])  # 只有前 5 个特征有效
y = X @ true_w + torch.randn(X.size(0)) * 0.3

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[-n_test:], y[-n_test:]

# 训练大模型（故意增加容量）
model = nn.Sequential(
    nn.Linear(20, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=20, min_delta=1e-4)

print("Early Stopping 训练过程:")
train_losses, val_losses = [], []

for epoch in range(500):
    model.train()
    pred = model(X_train).squeeze()
    train_loss = criterion(pred, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).squeeze()
        val_loss = criterion(val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    early_stopping(val_loss.item(), model)
    if early_stopping.should_stop:
        print(f"  Early Stopping at epoch {epoch+1}, val_loss={val_loss:.4f}")
        break

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# 恢复最佳模型
early_stopping.restore_best(model)
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze()
    test_loss = criterion(test_pred, y_test)
print(f"\n最佳模型测试损失: {test_loss:.4f}")

# 对比：不使用 Early Stopping 训练 500 轮
model2 = nn.Sequential(
    nn.Linear(20, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1)
)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

for epoch in range(500):
    pred = model2(X_train).squeeze()
    loss = criterion(pred, y_train)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

model2.eval()
with torch.no_grad():
    test_pred2 = model2(X_test).squeeze()
    test_loss2 = criterion(test_pred2, y_test)

print(f"无 Early Stopping 测试损失: {test_loss2:.4f}")
print(f"Early Stopping 改善了泛化: {test_loss2 > test_loss:.4f}")

# 过度训练与早停的权重范数对比
print(f"\n权重 L2 范数 - 早停: {sum(p.norm() for p in model.parameters()):.4f}")
print(f"权重 L2 范数 - 全训练: {sum(p.norm() for p in model2.parameters()):.4f}")
```

## 深度学习关联

- **实际训练的标准配置**：Early Stopping 几乎是所有深度学习训练的标配技术。它与学习率衰减配合使用——当验证损失 plateau 时，先尝试降低学习率继续训练，如果仍然没有改善则早停。这构成了一个鲁棒的训练终止策略。
- **计算资源的节约**：Early Stopping 不仅是正则化技术，也是节省计算资源的重要手段。在实际中大模型训练（如 BERT、GPT）可能需要数天甚至数周，提前检测到收敛可以节约大量计算成本。
- **双重正则化效果**：Early Stopping 可以与 L1/L2 正则化、Dropout 等其他正则化技术叠加使用。但由于 Early Stopping 本身已经是隐式正则化，同时使用多种正则化时需要注意总体的正则化强度，防止欠拟合。
