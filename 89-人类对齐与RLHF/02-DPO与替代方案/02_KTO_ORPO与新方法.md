# KTO、ORPO与新方法 - 人类对齐与RLHF

*探索 DPO 之外的人类对齐新范式：Kahneman-Tversky Optimization、Odds Ratio Preference Optimization、SimPO、自博弈方法及 Alignment Tax*

赔率（Odds）与赔率比（Odds Ratio）

```
赔率（Odds）：
  odds(y|x) = P(y|x) / (1 - P(y|x))
  即"发生概率 / 不发生概率"

赔率比（Odds Ratio）：
  OR(y_w, y_l | x) = odds(y_w|x) / odds(y_l|x)
  即"好回答的赔率 / 坏回答的赔率"

  当 OR > 1 时，y_w 比 y_l 更可能被生成
  ORPO 目标：最大化好回答相对于坏回答的赔率比

  逐 token 计算赔率：
  odds(y|x) = Π_t [p(y_t|x, y_
```

自博弈方法分类

| 方法 | 描述 | 代表工作 |
| --- | --- | --- |
| **SPIN** | 当前模型 vs SFT 模型的生成，通过自博弈消除幻觉 | SPIN (Chen et al., 2024) |
| **SPPO** | 自博弈生成偏好数据 + DPO 训练 | SPPO (Wu et al., 2024) |
| **IPO-Self** | 用自身不同温度采样的回答作为偏好对 | 理论分析 |
| **SPO** | 自我博弈偏好优化，两模型互相学习 | SPO 系列 |
| **RLAI Self-Play** | 类似 AlphaGo 的自我对弈范式 | DeepMind 研究 |

Alignment Tax 的表现形式

| 维度 | 下降表现 | 可能原因 |
| --- | --- | --- |
| **学术基准** | MMLU/GSM8K 等分数下降 | 对齐训练改变了知识表达 |
| **代码能力** | 代码生成能力退化 | 安全训练限制了代码输出 |
| **创造性** | 回答趋于保守、模板化 | 拒绝训练导致过度谨慎 |
| **多样性** | 输出多样性下降 | 偏好学习收敛到单一模式 |
| **多语言** | 非英语语言能力下降 | 偏好数据以英语为主 |

人类对齐方法全对比

| 方法 | 需要RM? | 需要参考模型? | 需要配对数据? | 训练阶段 | 复杂度 |
| --- | --- | --- | --- | --- | --- |
| **PPO (RLHF)** | 是 | 是 | 是（训练RM） | 3阶段 | 高 |
| **DPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **KTO** | 否 | 是 | **否** | 1阶段 | 低 |
| **ORPO** | 否 | **否** | 是 | 1阶段 | 最低 |
| **SimPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **IPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **SPIN** | 否 | 是 | **否** | 迭代 | 中 |


<!-- Converted from: 02_KTO_ORPO与新方法.html -->
