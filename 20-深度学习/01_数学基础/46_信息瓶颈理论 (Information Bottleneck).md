# 47_信息瓶颈理论 (Information Bottleneck)

## 核心概念

- **信息瓶颈 (Information Bottleneck, IB)**：一个关于表示学习的理论框架，目标是学习输入 $X$ 的一个压缩表示 $T$，使得 $T$ 在尽可能保留关于输出 $Y$ 的信息的同时，丢弃 $X$ 中与 $Y$ 无关的信息。
- **IB 目标函数**：
  $$\min_{p(t|x)} I(X; T) - \beta I(T; Y)$$
  其中 $\beta \geq 0$ 控制压缩与预测的权衡。$\beta$ 大时更关注预测（保留更多 $Y$ 信息），$\beta$ 小时更关注压缩（丢弃更多 $X$ 信息）。
- **信息平面 (Information Plane)**：以 $I(X; T)$ 为横轴、$I(T; Y)$ 为纵轴的二维平面。深度神经网络的学习过程可以在信息平面上画出轨迹。
- **压缩-预测权衡**：IB 框架的核心洞察——一个好的表示应该在压缩输入（减少 $I(X;T)$）和预测输出（增大 $I(T;Y)$）之间找到平衡。最优表示位于信息平面的"瓶颈"曲线上。
- **与率失真理论的关系**：IB 是率失真理论的有监督版本。率失真理论最小化 $I(X; \hat{X})$（压缩）同时约束 $\mathbb{E}[d(X, \hat{X})] \leq D$（保真度），而 IB 用 $I(T; Y)$ 代替了保真度约束。

## 数学推导

IB 的拉格朗日形式：
$$
\mathcal{L}[p(t|x)] = I(X; T) - \beta I(T; Y)
$$

其中互信息的定义：
$$
I(X; T) = \iint p(x, t) \log \frac{p(x, t)}{p(x)p(t)} dx dt = \iint p(t|x) p(x) \log \frac{p(t|x)}{p(t)} dx dt
$$
$$
I(T; Y) = \iint p(t, y) \log \frac{p(t, y)}{p(t)p(y)} dt dy
$$

自洽方程（IB 的迭代解）：
$$
p(t|x) = \frac{p(t)}{Z(x, \beta)} \exp\left(-\beta D_{\text{KL}}(p(y|x) \| p(y|t))\right)
$$
其中 $Z(x, \beta)$ 是归一化常数。

信息平面上的最优曲线：$I(T; Y)$ 作为 $I(X; T)$ 的函数是凹的，其导数为 $\beta^{-1}$。$\beta \to \infty$ 时 $T = Y$（最大压缩），$\beta = 0$ 时 $T = X$（无压缩）。

## 直观理解

- **"摘要"的隐喻**：信息瓶颈就像写文章摘要。$X$ 是原文，$T$ 是摘要，$Y$ 是读者的需求（关键信息）。好的摘要应该简短（压缩 $I(X;T)$）但保留所有关键点（增大 $I(T;Y)$）。$\beta$ 控制了"要压缩到什么程度"。
- **学习的两个阶段**：IB 理论认为深度神经网络的学习分为两个阶段：
  1. **拟合阶段 (Fitting Phase)**：$I(X;T)$ 和 $I(T;Y)$ 都快速增加，网络在"记忆"训练数据
  2. **压缩阶段 (Compression Phase)**：$I(X;T)$ 开始下降，而 $I(T;Y)$ 继续增加，网络在"忘记"与任务无关的细节
- **为什么压缩是有益的**：丢弃输入中的冗余信息可以防止过拟合——网络学习了"真正相关的模式"而非"数据的表面特征"。这类似于 Occam 剃刀原理：最简单的解释往往是最好的。

## 代码示例

```python
import numpy as np

# 1. 信息瓶颈的基本计算
# 离散变量的 IB 求解（Blahut-Arimoto 算法）

def ib_blahut_arimoto(p_xy, beta, n_iter=100):
    """
    求解信息瓶颈问题
    
    Args:
        p_xy: 联合分布 P(X, Y), shape (n_x, n_y)
        beta: 权衡参数
        n_iter: 迭代次数
    Returns:
        p_t_x: 编码器 P(T|X)
        p_y_t: 解码器 P(Y|T)
    """
    n_x, n_y = p_xy.shape
    n_t = n_x  # T 的基数等于 X 的基数
    
    # 边缘分布
    p_x = p_xy.sum(axis=1)
    p_y_given_x = p_xy / p_x[:, None]
    
    # 随机初始化
    p_t_x = np.random.dirichlet(np.ones(n_t), n_x).T  # P(t|x)
    
    for iteration in range(n_iter):
        # 计算边缘 P(T)
        p_t = p_t_x @ p_x
        
        # 计算 P(Y|T)
        p_y_t = np.zeros((n_y, n_t))
        for t in range(n_t):
            p_y_t[:, t] = p_y_given_x.T @ (p_t_x[t, :] * p_x) / p_t[t]
        
        # 计算 KL 散度
        kl_div = np.zeros((n_x, n_t))
        for t in range(n_t):
            for x in range(n_x):
                kl_div[x, t] = np.sum(p_y_given_x[x, :] * 
                    np.log((p_y_given_x[x, :] + 1e-10) / (p_y_t[:, t] + 1e-10)))
        
        # 更新 P(T|X)
        for x in range(n_x):
            logits = -beta * kl_div[x, :] + np.log(p_t + 1e-10)
            logits = logits - np.max(logits)  # 数值稳定
            p_t_x[:, x] = np.exp(logits) / np.sum(np.exp(logits))
    
    return p_t_x, p_y_t

# 测试简单情况
np.random.seed(42)
n_x, n_y = 10, 5
# 构造一个确定性关系
p_xy = np.random.rand(n_x, n_y)
p_xy = p_xy / p_xy.sum()

print("信息瓶颈 Blahut-Arimoto 算法:")
for beta in [0.1, 0.5, 1.0, 5.0]:
    p_t_x, p_y_t = ib_blahut_arimoto(p_xy, beta, n_iter=50)
    
    # 计算互信息
    p_x = p_xy.sum(axis=1)
    p_t = p_t_x @ p_x
    p_y = p_xy.sum(axis=0)
    
    # I(X;T)
    i_xt = 0
    for t in range(p_t_x.shape[0]):
        for x in range(len(p_x)):
            if p_t_x[t, x] > 0 and p_t[t] > 0 and p_x[x] > 0:
                i_xt += p_t_x[t, x] * p_x[x] * np.log(p_t_x[t, x] / p_t[t])
    
    # I(T;Y) 粗略估计
    i_ty = 0
    p_y_given_t = p_y_t.T  # we have this from the algorithm
    for t in range(p_t_x.shape[0]):
        for y in range(n_y):
            if p_y_given_t[t, y] > 0 and p_t[t] > 0 and p_y[y] > 0:
                i_ty += p_t[t] * p_y_given_t[t, y] * np.log(p_y_given_t[t, y] / p_y[y])
    
    print(f"  β={beta:.1f}: I(X;T)={i_xt:.4f}, I(T;Y)={i_ty:.4f}")

# 2. 信息平面轨迹模拟
print("\n信息平面轨迹 (不同 β):")
betas = np.logspace(-1, 1, 10)
i_xts = []
i_tys = []
for beta in betas:
    p_t_x, p_y_t = ib_blahut_arimoto(p_xy, beta, n_iter=30)
    p_x = p_xy.sum(axis=1)
    p_t = p_t_x @ p_x
    
    i_xt = 0
    for t in range(p_t_x.shape[0]):
        for x in range(len(p_x)):
            if p_t_x[t, x] > 0 and p_t[t] > 0 and p_x[x] > 0:
                i_xt += p_t_x[t, x] * p_x[x] * np.log(p_t_x[t, x] / p_t[t])
    i_xts.append(i_xt)
    
    # 近似 I(T;Y) = I(X;Y) - I(X;Y|T)
    p_y = p_xy.sum(axis=0)
    i_xy = 0
    for x in range(n_x):
        for y in range(n_y):
            if p_xy[x, y] > 0:
                i_xy += p_xy[x, y] * np.log(p_xy[x, y] / (p_x[x] * p_y[y]))
    i_tys.append(i_xy - 0.1)  # 近似

print(f"  I(X;Y) = {i_xy:.4f}")
for b, ixt, ity in zip(betas, i_xts, i_tys):
    print(f"  β={b:.2f}: I(X;T)={ixt:.3f}, I(T;Y)={ity:.3f}")

# 3. 压缩与预测的权衡
print("\n信息瓶颈的直觉:")
print("  β→∞: 极端压缩 (T退化为常数), I(X;T)=0, I(T;Y)=0")
print("  β=0: 无压缩 (T=X), I(X;T)最大, I(T;Y)最大")
print("  最优β: 在I(X;T)和I(T;Y)间取得平衡")
```

## 深度学习关联

- **信息瓶颈与深度网络的可解释性**：IB 理论（Tishby 等 2015）提出深度网络训练经历"拟合-压缩"两阶段。后续研究发现，使用 ReLU 激活的网络不一定出现明显的压缩阶段，但信息平面分析仍是理解深度学习动态的强大视角，可以揭示网络何时开始过拟合。
- **变分信息瓶颈 (VIB)**：Alemi 等人将 IB 扩展到深度神经网络，提出变分信息瓶颈：
  $$\mathcal{L}_{\text{VIB}} = \mathbb{E}[\log p(y|z)] - \beta D_{\text{KL}}(p(z|x) \| p(z))$$
  其中 $z$ 是中间表示的噪声版本。VIB 在多个任务上提高了模型的泛化能力和对抗鲁棒性。IB 思想的这一实用化版本与 VAE 有密切联系。
- **对比学习与互信息最大化**：SimCLR、MoCo 等对比学习方法可以看作 IB 框架的实例——目标是在压缩增强不变信息的同时最大化与语义标签的互信息。InfoNCE 损失函数是互信息的下界估计器，其优化相当于在信息平面上"向右上"方向移动。
- **信息平面与过拟合检测**：实践中，$I(X;T)$ 和 $I(T;Y)$ 可以作为监控训练进度的指标。当 $I(T;Y)$ 在验证集上开始下降而 $I(X;T)$ 继续增加时，说明模型正在过拟合。这种信息论视角为早停（early stopping）提供了理论依据。
