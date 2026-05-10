# PPO在RLHF中的应用 - 人类对齐与RLHF

*深入理解 Proximal Policy Optimization 算法在 RLHF 中的具体实现，涵盖 Clipped Surrogate 目标、GAE 优势估计、KL 惩罚及完整 Pipeline*

KL 惩罚在 RLHF 中的三种实现方式

| 方式 | 公式 | 特点 |
| --- | --- | --- |
| **固定系数 KL** | r = r_RM - β * KL | 简单直接，但 β 难以调优 |
| **自适应 KL** | 根据实际 KL 调整 β | 自动调节，InstructGPT 使用 |
| **KL 预算** | 设置 KL 上限 | 约束总偏离量 |

PPO-RLHF 关键超参数

| 超参数 | InstructGPT 值 | 作用 | 调优建议 |
| --- | --- | --- | --- |
| lr (PPO) | 9.65e-6 | 策略更新学习率 | 从 1e-6 开始，过大会崩溃 |
| clip_epsilon | 0.2 | 裁剪范围 | 0.1-0.3，越小越保守 |
| kl_coef (β) | 自适应, 目标 6.0 | KL 惩罚强度 | 用自适应控制器 |
| gamma | 1.0 | 折扣因子 | RLHF 中通常用 1.0 |
| gae_lambda | 0.95 | GAE 偏差-方差权衡 | 0.9-0.99 常用 |
| ppo_epochs | 1 | 每批数据重复训练次数 | 1-4，过多会过拟合 |
| vf_coef | 0.5 | Value loss 权重 | 0.1-1.0 |
| entropy_coef | 0.01 | 熵正则化 | 过小则模式坍塌 |
| batch_size | 512 prompts | 每步 PPO 使用的 prompt 数 | 越大越稳定 |
| rollout_size | 1 回答/prompt | 每个 prompt 生成的回答数 | 1-4 |

PPO-RLHF 开源框架

| 框架 | 特点 | 适用场景 |
| --- | --- | --- |
| **TRL (HuggingFace)** | 集成度高，API 友好，与 HF 生态无缝对接 | 中小规模实验 |
| **DeepSpeed-Chat** | ZeRO 优化，显存效率高，支持大规模训练 | 百亿级模型训练 |
| **OpenRLHF** | 高性能，支持 Ray 分布式，解耦 4 个模型 | 大规模生产训练 |
| **trlX** | 支持分布式 PPO，设计灵活 | 研究实验 |


<!-- Converted from: 02_PPO在RLHF中的应用.html -->
