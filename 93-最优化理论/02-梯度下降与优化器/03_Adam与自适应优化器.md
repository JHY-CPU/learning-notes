# Adam与自适应优化器 — 最优化理论笔记


## 一、AdaGrad：自适应学习率的开端


### 1.1 核心思想


不同参数需要不同的学习率：频繁更新的参数应该用较小的学习率，稀疏参数应该用较大的学习率。


### 1.2 算法公式


累积历史梯度的平方：


$$
\(s_t = s_{t-1} + g_t \odot g_t\)
                \(x_{t+1} = x_t - \frac{\alpha}{\sqrt{s_t + \epsilon}} \odot g_t\)
$$


其中 \( g_t = \nabla f(x_t) \)，\( \odot \) 为逐元素乘法，\( \epsilon \approx 10^{-8} \) 防止除零。


### 1.3 优缺点


- **优点：**
   自动调节每个参数的学习率，适合稀疏数据（如NLP的词嵌入）
- **致命缺点：**
   \( s_t \) 单调递增且不衰减，导致学习率持续缩小至接近零，训练提前停止


## 二、RMSProp：解决AdaGrad的衰减问题


### 2.1 核心改进


Hinton在2012年提出：用**指数移动平均**替代累积求和，使旧梯度信息逐渐"遗忘"：


$$
\(s_t = \rho s_{t-1} + (1-\rho) g_t \odot g_t\)
                \(x_{t+1} = x_t - \frac{\alpha}{\sqrt{s_t + \epsilon}} \odot g_t\)
$$


其中 \( \rho \) 通常取 0.9 或 0.99。这个衰减因子使得学习率不会无限缩小。


### 2.2 与AdaGrad的对比


| 特性 | AdaGrad | RMSProp |
| --- | --- | --- |
| 梯度累积方式 | 累加和（无衰减） | 指数移动平均 |
| 学习率趋势 | 单调递减至0 | 有界，不会消失 |
| 适用场景 | 凸问题、稀疏梯度 | 非凸问题、深度学习 |
| 收敛性 | 理论上保证 | 实践效果好，理论分析复杂 |


> **Note:** **注：**
> RMSProp没有正式论文，来自Hinton的Coursera课程讲义，但在实践中极为有效。


## 三、Adam = Momentum + RMSProp


### 3.1 Adam的核心思想


Adam（Adaptive Moment Estimation，Kingma & Ba, 2015）同时维护两个状态：


- **一阶矩估计 \( m_t \)**
   ：梯度的指数移动平均（类似Momentum）
- **二阶矩估计 \( v_t \)**
   ：梯度平方的指数移动平均（类似RMSProp）


### 3.2 Adam完整公式


$$
一阶矩（动量）：
                \(m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t\)
二阶矩（自适应学习率）：
                \(v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2\)
偏差修正：
                \(\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}\)
参数更新：
                \(x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t\)
$$


### 3.3 偏差修正的重要性


由于 \( m_0 = 0, v_0 = 0 \)，在训练初期 \( m_t, v_t \) 会系统性地偏小（接近零）。偏差修正解决了这个问题：


- 当 \( t=1, \beta_1=0.9 \) 时：\( m_1 = 0.1g_1 \)，但 \( \hat{m}_1 = g_1 \)（修正后）
- 当 \( t \to \infty \) 时：\( \beta_1^t \to 0 \)，修正因子趋近于1，无影响


### 3.4 默认超参数


| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| \( \alpha \) | 0.001 | 学习率 |
| \( \beta_1 \) | 0.9 | 一阶矩衰减率（梯度动量） |
| \( \beta_2 \) | 0.999 | 二阶矩衰减率（梯度平方动量） |
| \( \epsilon \) | 10^-8^ | 数值稳定项 |


## 四、AdamW：权重衰减与梯度解耦


### 4.1 标准Adam中L2正则化的问题


在标准Adam中加入L2正则化 \( \lambda\|x\|^2 \) 时，权重衰减项 \( \lambda x \) 也参与了自适应学习率的缩放：


$$
\(g_t^{\text{total}} = \nabla f(x_t) + \lambda x_t\)
$$


这导致权重衰减在自适应学习率下表现不一致：某些参数的衰减被放大，某些被缩小。


### 4.2 AdamW的解耦方案


$$
\(x_{t+1} = x_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda x_t\right)\)
$$


权重衰减项 \( \lambda x_t \) 直接乘以学习率 \( \alpha \)，不参与自适应缩放。


> **Important:** **实践结论：**
> Loshchilov & Hutter (2019) 证明AdamW在几乎所有实验中都优于Adam+L2。这是目前训练Transformer等模型的标准选择。
>
>
> PyTorch:
> `torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`


## 五、各优化器对比


| 优化器 | 核心机制 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| SGD | 纯梯度 | 简单、泛化好 | 震荡、收敛慢 | CV调优后期 |
| SGD+Momentum | 梯度累积 | 抑制震荡 | 学习率需精心调 | ResNet等CV模型 |
| AdaGrad | 累积平方梯度 | 自适应、稀疏友好 | 学习率衰减至0 | 稀疏特征 |
| RMSProp | 指数移动平均平方梯度 | 自适应、不衰减 | 无收敛保证 | RNN |
| Adam | 一阶+二阶矩估计 | 自适应、收敛快 | 泛化可能不如SGD | 通用默认选择 |
| AdamW | Adam + 解耦权重衰减 | 正则化效果更好 | 多一个超参 | Transformer/NLP |
| LAMB | 层级自适应学习率 | 大batch训练 | 实现复杂 | 大batch BERT |


## 六、PyTorch代码示例


```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 构建简单模型和数据
# ==========================================
torch.manual_seed(42)

# 生成数据
n_samples = 1000
X = torch.randn(n_samples, 20)
true_w = torch.zeros(20)
true_w[[0, 5, 10]] = [2.0, -1.5, 3.0]
y = X @ true_w + 0.1 * torch.randn(n_samples)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 简单线性模型
def create_model():
    return nn.Linear(20, 1)

# ==========================================
# 定义训练函数
# ==========================================
def train_model(optimizer_name, model, epochs=50):
    criterion = nn.MSELoss()
    losses = []

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.05)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses

# ==========================================
# 对比各优化器
# ==========================================
optimizers = ['SGD', 'SGD+Momentum', 'Adam', 'AdamW']
results = {}

for opt_name in optimizers:
    model = create_model()
    losses = train_model(opt_name, model)
    results[opt_name] = losses
    print(f"{opt_name:18s} | 最终损失: {losses[-1]:.6f}")

# ==========================================
# 学习率调度示例
# ==========================================
print("\n=== 学习率调度 ===")
model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# StepLR
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
print(f"StepLR:   初始lr={optimizer.param_groups[0]['lr']:.4f}")
for epoch in range(25):
    scheduler1.step()
    if epoch % 5 == 4:
        print(f"  Epoch {epoch+1}: lr={optimizer.param_groups[0]['lr']:.6f}")

# CosineAnnealing
model2 = create_model()
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
print(f"\nCosineAnnealing:")
for epoch in [0, 10, 24, 36, 49]:
    optimizer2.param_groups[0]['lr'] = 0.01  # 重置
    scheduler2_temp = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
    for _ in range(epoch + 1):
        scheduler2_temp.step()
    print(f"  Epoch {epoch+1}: lr={optimizer2.param_groups[0]['lr']:.6f}")
```


## 七、选择建议


- **默认首选 Adam 或 AdamW：**
   收敛快，超参数鲁棒性好
- **CV任务（如ResNet）：**
   SGD + Momentum + CosineAnnealing 往往泛化更好
- **NLP/Transformer：**
   AdamW 是标准选择
- **追求极致性能：**
   先用Adam快速收敛，后切SGD微调（learning rate warm restart）
- **学习率比优化器选择更重要：**
   一个调好的SGD可能比默认Adam好


> **Note:** **总结：**
> 自适应优化器通过维护梯度的一阶和二阶矩估计，为不同参数自动调整学习率。Adam是最流行的优化器，AdamW解决了权重衰减与自适应学习率的耦合问题。选择优化器时应结合任务类型、模型架构和调参经验。


<!-- Converted from: 03_Adam与自适应优化器.html -->
