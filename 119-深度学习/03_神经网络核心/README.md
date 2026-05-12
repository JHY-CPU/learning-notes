# 神经网络核心

## 一、前馈神经网络

### 1.1 网络结构

前馈神经网络（Feedforward Neural Network）由输入层、隐藏层和输出层组成，信息单向流动。

**第 $l$ 层的计算：**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f(z^{(l)})$$

其中 $W$ 为权重矩阵，$b$ 为偏置向量，$f$ 为激活函数。

### 1.2 激活函数

| 函数 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | 输出(0,1)，易梯度消失 |
| Tanh | $\tanh(x)$ | 输出(-1,1)，零中心 |
| ReLU | $\max(0, x)$ | 计算快，可能神经元死亡 |
| LeakyReLU | $\max(\alpha x, x)$ | 缓解死亡ReLU |
| GELU | $x \cdot \Phi(x)$ | Transformer常用 |
| Swish | $x \cdot \sigma(\beta x)$ | 平滑，性能好 |

### 1.3 万能近似定理

一个包含足够多隐藏单元的单隐层前馈网络，可以以任意精度近似任何连续函数。

---

## 二、反向传播

### 2.1 计算图

反向传播基于链式法则，沿着计算图反向传播梯度。

**前向传播**：从输入到输出计算损失
**反向传播**：从损失到参数计算梯度

### 2.2 梯度计算

对于损失 $L$：
$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot (a^{(l-1)})^T$$
$$\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial z^{(l)}}$$
$$\frac{\partial L}{\partial a^{(l-1)}} = (W^{(l)})^T \cdot \frac{\partial L}{\partial z^{(l)}}$$

### 2.3 梯度问题

- **梯度消失**：深层网络中梯度指数级衰减（Sigmoid/Tanh导致）
- **梯度爆炸**：梯度指数级增长
- **解决方法**：
  - ReLU激活函数
  - 残差连接（ResNet）
  - Batch Normalization
  - 梯度裁剪（Gradient Clipping）
  - 合理的权重初始化（Xavier/He初始化）

---

## 三、优化算法

### 3.1 梯度下降变体

| 算法 | 更新规则 | 特点 |
|------|----------|------|
| SGD | $\theta \leftarrow \theta - \eta \nabla L$ | 基础，可能震荡 |
| SGD+Momentum | $v \leftarrow \beta v + \eta \nabla L; \theta \leftarrow \theta - v$ | 加速收敛 |
| Adagrad | 自适应学习率 | 适合稀疏数据 |
| RMSProp | 指数移动平均 | 解决Adagrad衰减问题 |
| Adam | 动量+自适应 | 最常用 |
| AdamW | 解耦权重衰减 | Transformer标配 |

### 3.2 Adam优化器

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$（一阶矩估计）
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$（二阶矩估计）
$$\hat{m}_t = m_t / (1-\beta_1^t)$$（偏差修正）
$$\hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

默认超参数：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$

---

## 四、正则化技术

### 4.1 Dropout

训练时以概率 $p$ 随机将神经元输出置零，测试时使用全部神经元并缩放。

- 作用：防止共适应，相当于模型集成
- 典型值：隐藏层 $p=0.5$，输入层 $p=0.2$

### 4.2 Batch Normalization

对每个mini-batch的激活值进行标准化：
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

作用：加速训练、允许更大学习率、轻微正则化效果

### 4.3 权重初始化

- **Xavier/Glorot**：$W \sim \mathcal{N}(0, 2/(n_{in}+n_{out}))$，适合Sigmoid/Tanh
- **He初始化**：$W \sim \mathcal{N}(0, 2/n_{in})$，适合ReLU
- **原则**：保持各层激活值方差一致

---

## 五、损失函数

| 任务 | 损失函数 | 公式 |
|------|----------|------|
| 二分类 | Binary CE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| 多分类 | Categorical CE | $-\sum y_i \log\hat{y}_i$ |
| 回归 | MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| 回归 | MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |
| 对比 | Triplet Loss | $\max(0, d(a,p) - d(a,n) + \alpha)$ |
