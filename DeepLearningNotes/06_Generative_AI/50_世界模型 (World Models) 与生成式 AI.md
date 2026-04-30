# 50_世界模型 (World Models) 与生成式 AI

## 核心概念
- **世界模型 (World Model)**：对物理世界（或特定环境）的内部表征，能够预测"如果执行某个动作，世界会如何演变"。它是对环境动力学的生成式模拟器。
- **World Models 的核心三要素**：(1) 感知模块（Vision Model）——将感官输入压缩为潜在状态，(2) 记忆/RNN 模块——维护跨时间步的状态，(3) 决策/控制模块——基于预测选择动作。
- **Dreamer / DreamerV2 / DreamerV3**：DeepMind 的系列工作——在潜在空间中使用强化学习训练世界模型，智能体在"梦境"中规划和学习，而不需要实时与环境交互。
- **生成式世界模型 (Generative World Model)**：不仅预测状态的下一帧，还能生成完整的未来帧序列——这实质上是一个条件生成模型，以当前状态和动作为条件生成未来观测。
- **基于扩散的世界模型 (UniSim, Genie)**：使用扩散模型作为世界模型的预测引擎——给定当前帧和动作，扩散模型生成下一帧。这种框架能生成高度逼真的未来帧预测。
- **规划 (Planning)**：世界模型的核心应用——通过在"想象中的未来"中尝试多种动作序列，选择最优路径。这类似于人类的"思考和预演"能力。
- **开环 vs 闭环**：开环预测（给定真值帧预测下一帧）和闭环预测（用预测的帧预测再下一帧）是世界模型评估的重要区分——闭环中误差会累积。

## 数学推导

**世界模型的状态空间模型**：

世界模型通常建模为部分可观测马尔可夫决策过程 (POMDP)：

$$
\text{编码: } z_t \sim q_\phi(z_t | o_{1:t}, a_{1:t-1})
$$

$$
\text{转移: } z_t \sim p_\phi(z_t | z_{t-1}, a_{t-1})
$$

$$
\text{重建: } o_t \sim p_\phi(o_t | z_t)
$$

其中 $o_t$ 是观测，$a_t$ 是动作，$z_t$ 是潜在状态。

**Dreamer 的损失函数**：

1. 感知模块（VAE 风格）：

$$
\mathcal{L}_{\text{vision}} = -\log p_\phi(o_t | z_t) + KL(q_\phi(z_t | o_t) \parallel p(z_t))
$$

2. 转移模块（预测状态）：

$$
\mathcal{L}_{\text{transition}} = KL(q_\phi(z_t | o_t) \parallel p_\phi(z_t | z_{t-1}, a_{t-1}))
$$

3. 策略（在梦境中学习）：

$$
\mathcal{L}_{\text{policy}} = -\mathbb{E}_{z_{1:H} \sim p_\phi, a_{1:H} \sim \pi_\theta} \left[ \sum_{t=1}^H R_\psi(z_t, a_t) \right]
$$

**基于扩散的预测**：

给定当前帧 $x_t$ 和动作 $a_t$，扩散模型预测下一帧 $x_{t+1}$：

$$
x_{t+1} = \text{DDIM}(\epsilon_\theta(\cdot | x_t, a_t), z_T)
$$

训练损失：

$$
\mathcal{L} = \mathbb{E}_{x_t, x_{t+1}, a_t, \epsilon} [\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_{t+1} + \sqrt{1-\bar{\alpha}_t} \epsilon, x_t, a_t)\|^2]
$$

## 直观理解
- **世界模型 = 梦境中的模拟器**：人类可以在脑海中"预演"一个场景——"如果我现在向左转，会碰到椅子吗？"世界模型给了 AI 同样的能力——在"想象"中进行试错和学习，不需要在现实世界中冒险。
- **Why Dreamer 有效**：就像你学习骑自行车一样——不需要真的摔 100 次才知道怎么平衡。世界模型构建了一个心理模拟器，在"想象的场景"中训练骑车技巧。
- **生成式世界模型 = 可以预知未来的生成 AI**：普通生成模型（如 Stable Diffusion）生成的是静止的、无条件的图像。生成式世界模型生成的是"如果做某个动作，世界会变成什么样"——这是预测性的生成。
- **开环 vs 闭环的困境**：开环预测就像只看下一张牌——容易；闭环预测就像预言整局牌——误差逐轮放大。优秀的世界模型能保持长时间稳定的闭环预测。
- **世界模型 vs 视频生成**：视频生成只关心视觉质量，世界模型关心预测正确性（物理一致性）。一辆车在视频中消失又出现可能对视频生成无妨，但对世界模型是致命的（违反物理常识）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    """
    简化的世界模型
    
    感知编码器 + 状态转移 + 预测解码器
    """
    def __init__(self, obs_dim=64*64*3, action_dim=4, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 感知编码器（观测 -> 潜状态）
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),  # mu + logvar
        )
        
        # 状态转移模型（潜状态 + 动作 -> 下一潜状态）
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),  # mu + logvar
        )
        
        # 预测解码器（潜状态 -> 预测观测）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, obs_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, obs):
        h = self.encoder(obs)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar
    
    def predict_next_state(self, z, action):
        """预测下一步的潜状态"""
        z_action = torch.cat([z, action], dim=-1)
        h = self.transition(z_action)
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z_next = mu + std * torch.randn_like(std)
        return z_next, mu, logvar
    
    def decode(self, z):
        """从潜状态重建观测"""
        return self.decoder(z)
    
    def forward(self, obs, action, next_obs):
        # 编码当前观测
        z, mu_z, logvar_z = self.encode(obs)
        
        # 预测下一潜状态
        z_pred, mu_pred, logvar_pred = self.predict_next_state(z, action)
        
        # 解码预测
        obs_pred = self.decode(z_pred)
        
        # 编码真实下一观测（用于 KL 散度）
        z_next, mu_next, logvar_next = self.encode(next_obs)
        
        # 损失：预测误差 + KL 散度
        recon_loss = F.mse_loss(obs_pred, next_obs)
        # 预测分布与真实编码分布的 KL
        kl_loss = self._kl_divergence(mu_pred, logvar_pred, mu_next, logvar_next)
        
        return recon_loss + kl_loss
    
    def _kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """两个高斯分布的 KL 散度"""
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        return 0.5 * torch.mean(
            logvar2 - logvar1 + (var1 + (mu1 - mu2)**2) / var2 - 1
        )
    
    @torch.no_grad()
    def imagine_future(self, init_obs, actions, temperature=1.0):
        """在"梦境"中想象未来"""
        z, _, _ = self.encode(init_obs)
        
        predictions = []
        for action in actions:
            z_pred, _, _ = self.predict_next_state(z, action)
            obs_pred = self.decode(z_pred)
            predictions.append(obs_pred)
            z = z_pred  # 角色循环
        
        return torch.stack(predictions)

# 在梦境中进行规划
def plan_in_dreams(world_model, current_obs, candidate_actions, horizon=10):
    """
    在世界模型的梦境中规划最优动作序列
    """
    B = current_obs.size(0)
    n_candidates = candidate_actions.size(1)
    
    # 编码当前状态
    z, _, _ = world_model.encode(current_obs)
    
    best_score = -float('inf')
    best_plan = None
    
    for i in range(n_candidates):
        z_t = z
        total_reward = 0
        
        for t in range(horizon):
            action = candidate_actions[:, i, t]
            # 预测下一状态
            z_pred, _, _ = world_model.predict_next_state(z_t, action)
            obs_pred = world_model.decode(z_pred)
            
            # 计算奖励（使用模拟奖励函数）
            reward = simulate_reward(obs_pred)
            total_reward += reward
            
            z_t = z_pred
        
        if total_reward > best_score:
            best_score = total_reward
            best_plan = candidate_actions[:, i]
    
    return best_plan, best_score

def simulate_reward(obs):
    """模拟奖励函数（在实际中使用预训练奖励模型）"""
    return torch.randn(obs.size(0)).mean().item()

print("=== 世界模型 (World Models) 与生成式 AI ===")
print()

# 模拟
obs_dim = 64 * 64 * 3
model = WorldModel(obs_dim=obs_dim, action_dim=4, latent_dim=256)

obs = torch.randn(4, obs_dim)
action = torch.randn(4, 4)
next_obs = torch.randn(4, obs_dim)

loss = model(obs, action, next_obs)
print(f"世界模型训练损失: {loss.item():.4f}")
print(f"潜状态维度: {model.latent_dim}")
print(f"世界模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print()

print("世界模型的能力:")
print("1. 状态编码: 观测 -> 潜状态")
print("2. 状态转移: 预测下一步潜状态")
print("3. 观测重建: 潜状态 -> 可理解的观测")
print("4. 在梦境中规划: 想象未来并优化决策")
print()
print("生成式 AI 与世界模型的交叉点:")
print("- 扩散模型作为世界模型的预测引擎 (UniSim)")
print("- 视频生成模型作为世界模型的视觉基础")
print("- 世界模型为生成内容提供物理一致性约束")
```

## 深度学习关联
- **DayDreamer / DreamerV3**：世界模型在机器人领域的应用——机器人在世界模型的"梦境"中学会行走、抓取等技能，然后将策略部署到真实机器人。DreamerV3 在 Minecraft 等复杂环境中也取得了优异成绩。
- **UniSim (Universal Simulator)**：用大规模视频数据和文本指令训练一个"通用模拟器"——给定当前帧和描述性指令（"打开门"），生成下一帧。这实质上是一个以动作为条件的生成式世界模型。
- **Genie (Google DeepMind)**：从互联网视频中无监督学习世界模型——学会了 2D 平台游戏的环境动力学，玩家可以给智能体一个"目标帧"，Genie 通过世界模型找到达到目标帧的动作序列。
- **SORA (OpenAI)**：视频生成模型，内部可能融合了世界模型的理念——生成视频时不仅关注视觉质量，还表现出对物理规律的理解（如物体碰撞、光影变化、流体运动），被称为"世界模拟器"。
