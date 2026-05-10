# DPO原理与实现 - 人类对齐与RLHF

*Direct Preference Optimization——无需 Reward Model 和 RL 训练的偏好对齐方法，从 RLHF 到 closed-form 解的完整数学推导*

RLHF 优化问题

```
max_π E_{x~D, y~π(y|x)}[r(x,y)] - β * D_KL(π(y|x) || π_ref(y|x))

展开 KL 散度：
= max_π E[r(x,y)] - β * E[log π(y|x) - log π_ref(y|x)]

= max_π E_{x,y}[r(x,y)/β + log π_ref(y|x) - log π(y|x)]

这等价于最小化：
= min_π E_{x,y}[log π(y|x) - (r(x,y)/β + log π_ref(y|x))]

令 Z(x) = Σ_y π_ref(y|x) * exp(r(x,y)/β)（配分函数）

可以证明最优策略为：
π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)
```

Reward 反解

```
由 π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)

两边取 log：
log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)

整理得：
r(x,y) = β * [log π*(y|x) - log π_ref(y|x)] + β * log Z(x)

关键：配分函数 Z(x) 不依赖于 y！
因此在比较两个回答时，Z(x) 会消掉。
```

DPO 损失函数推导

```
P(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))

代入 r(x,y) = β * [log π(y|x) - log π_ref(y|x)] + β*log Z(x)：

= σ(β * [log π(y_w|x) - log π_ref(y_w|x) - log π(y_l|x) + log π_ref(y_l|x)])

= σ(β * log[π(y_w|x)/π_ref(y_w|x)] - β * log[π(y_l|x)/π_ref(y_l|x)])

定义隐式 Reward：
r_θ(x,y) = β * log[π_θ(y|x) / π_ref(y|x)]

DPO 损失（负对数似然）：
L_DPO(θ) = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

这就是 DPO 的最终形式——一个简单的二元交叉熵损失！
只需要策略模型 π_θ 和冻结的参考模型 π_ref，无需 RM 和 RL。
```

β 值的影响分析

| β 值 | 行为 | 适用场景 |
| --- | --- | --- |
| **β → 0** | 模型大幅偏向 chosen、远离 rejected | 偏好明确、数据质量高 |
| **β = 0.01 ~ 0.05** | 较强的偏好学习 | Chat 模型微调（常用） |
| **β = 0.1 ~ 0.2** | 温和的偏好学习 | 一般对齐任务 |
| **β → ∞** | 模型保持不变（π_θ ≈ π_ref） | 不做对齐 |

DPO vs PPO 深度对比

| 维度 | PPO (RLHF) | DPO |
| --- | --- | --- |
| **训练范式** | RM 训练 + RL 优化（两阶段） | 单阶段监督学习 |
| **模型数量** | 4 个（策略/参考/RM/Value Head） | 2 个（策略/参考） |
| **训练稳定性** | 不稳定，超参敏感 | 稳定，像正常微调 |
| **计算成本** | 高（需要在线采样 + RM 推理） | 低（离线数据 + 简单 loss） |
| **数据需求** | 需要在线生成数据 | 离线偏好对即可 |
| **探索能力** | 可以探索新的输出空间 | 受限于训练数据分布 |
| **分布外泛化** | 较好（在线采样覆盖更多分布） | 可能较差（离线数据有限） |
| **超参数** | 多（lr, clip, kl_coef, gae_lambda...） | 少（lr, beta） |
| **实现难度** | 高（需要工程优化） | 低（几行核心代码） |
| **最佳效果** | 理论上界更高 | 实践中常与 PPO 持平 |
| **适用场景** | 大规模生产训练 | 快速迭代、资源有限 |

DPO 家族方法对比

| 方法 | 损失函数 | 核心改进 |
| --- | --- | --- |
| **DPO** | -log σ(β * Δ) | Baseline，简单有效 |
| **IPO** | (Δ - 1/(2β))^2 | L2 损失防过拟合 |
| **cDPO** | 带 Label Smoothing 的 DPO | 噪声鲁棒性 |
| **SPPO** | 自博弈生成 + DPO | 无需人类标注 |
| **EXO** | Expert Optimization | 扩展到多专家 |
| **RDPO** | Regularized DPO | 更强正则化 |
| **Cal-DPO** | 校准的 DPO | 概率校准 |


<!-- Converted from: 01_DPO原理与实现.html -->
