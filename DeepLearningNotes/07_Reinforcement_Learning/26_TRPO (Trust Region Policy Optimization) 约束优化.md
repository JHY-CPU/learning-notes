# 26_TRPO (Trust Region Policy Optimization) 约束优化

## 核心概念

- **TRPO (Trust Region Policy Optimization)**：2015 年 Schulman 等人提出的策略优化算法。通过在每一步更新中强制执行 KL 散度约束，确保策略变化不会太大，从而保证更新的单调改进。
- **信任区域 (Trust Region)**：在参数空间中划出一个"安全区域"，在该区域内使用局部近似来优化目标函数。这借鉴了优化理论中的置信域方法（trust region methods）。
- **KL 散度约束**：$\mathbb{E}_t[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \le \delta$，其中 $\delta$ 是信任区域半径（通常 0.01 到 0.001）。约束确保新旧策略在**所有状态**上的分布差异不要太大。
- **替代目标 (Surrogate Objective)**：TRPO 使用重要性采样构造的替代目标函数，在信任区域约束下最大化该函数。这个替代目标与真实目标在局部相匹配。
- **共轭梯度法 (Conjugate Gradient)**：TRPO 使用共轭梯度法高效近似计算自然梯度方向，避免了直接计算大矩阵的逆（Fisher 信息矩阵的逆），使得算法在大型神经网络上可行。
- **单调改进保证**：TRPO 理论上保证了每次迭代都能获得单调的策略改进（直到收敛），这是通过"替代目标 + KL 约束"的理论框架实现的。

## 数学推导

$$
\text{TRPO 优化问题: }
$$

$$
\max_\theta \mathbb{E}_{s \sim d_{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A_{\pi_{\text{old}}}(s, a) \right]
$$

$$
\text{s.t. } \mathbb{E}_{s \sim d_{\pi_{\text{old}}}} \left[ \text{KL}\left( \pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s) \right) \right] \le \delta
$$

$$
\text{自然梯度近似: } \Delta\theta \approx \alpha \, F^{-1} \, g
$$

$$
\text{其中: } F = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \, \nabla_\theta \log \pi_\theta(a|s)^T \right] \quad \text{(Fisher 信息矩阵)}
$$

$$
g = \nabla_\theta \mathbb{E} \left[ \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A \right] \quad \text{(策略梯度)}
$$

$$
\text{通过共轭梯度求解: } F^{-1} g \approx x \quad \text{使得 } Fx = g
$$

**推导说明**：
- TRPO 的理论基础是"保守策略迭代"（conservative policy iteration），通过 $\eta(\pi) \ge L_\pi(\tilde{\pi}) - C \cdot \text{KL}_{\text{max}}$ 给出了策略改进的下界。
- 自然梯度 $F^{-1}g$ 不同于标准梯度 $g$——它考虑了参数空间的 Riemannian 几何结构，使更新方向在策略分布空间中更直接。
- 共轭梯度法将求逆计算复杂度从 $O(N^3)$ 降到 $O(N^2)$，其中 $N$ 是参数量。每步只需要 $N \times k$ 次操作（$k$ 是共轭梯度迭代次数）。

## 直观理解

TRPO 就像一个"安全第一"的登山者：

想象你在浓雾中爬一座山（优化策略参数 $\theta$）：
- **标准策略梯度（爬山）**：用登山杖探一下脚下的坡度（计算梯度），然后朝最陡的方向迈一大步。但如果脚下的坡度突然变化（梯度估计噪声大），你可能一脚踩空滚下山。
- **TRPO（安全登山）**：每走一步前，先用脚仔细探明前方 1 米范围内的地形（信任区域），确保在这个区域内地形是可信的（目标函数近似准确），然后迈出在这个"安全区域"内最好的一步。如果前面是悬崖，TRPO 会停下来，因为它知道"超出这个范围的地形我不确定"。

**为什么需要信任区域？**
一个类比是"修路"和"开车"的区别。如果你在崎岖的山路上开车（策略参数空间复杂），你的方向盘不能每次转动 90 度——虽然梯度告诉你"前方"（局部方向）可以开，但大幅转向可能直接翻车。信任区域确保你每个调整都是小幅的、可控的。

**TRPO vs PPO**：
- TRPO 是"精密的工程计算"——每一步都精确计算信任区域边界（共轭梯度+线搜索）。
- PPO 是"便宜的保险杠"——不精确计算边界，但用裁剪确保不出界。
- TRPO 更安全（有硬性约束保证），PPO 更简单（工程实现友好）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

class TRPOAgent:
    """简化版 TRPO 实现（核心逻辑）"""
    def __init__(self, state_dim, action_dim, hidden=64, delta=0.01, gamma=0.99):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.delta = delta  # KL 约束阈值
        self.gamma = gamma
    
    def get_action_and_log_prob(self, state):
        features = self.policy(state)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist
    
    def get_kl_divergence(self, state, old_dist):
        """计算当前策略与旧策略的 KL 散度"""
        features = self.policy(state)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        new_dist = torch.distributions.Normal(mean, std)
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).sum(dim=-1)
        return kl.mean()
    
    def flat_params(self, params_list):
        """将参数展平为一维向量"""
        return torch.cat([p.data.view(-1) for p in params_list])
    
    def set_params(self, params_list, flat_params):
        """从一维向量恢复参数"""
        idx = 0
        for p in params_list:
            n = p.numel()
            p.data.copy_(flat_params[idx:idx+n].view(p.shape))
            idx += n
    
    def conjugate_gradient(self, A, b, nsteps=10, residual_tol=1e-10):
        """共轭梯度法求解 Ax = b"""
        x = torch.zeros_like(b)
        r = b - A(x)
        p = r.clone()
        rdotr = r.dot(r)
        
        for i in range(nsteps):
            Ap = A(p)
            alpha = rdotr / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def update(self, states, actions, advantages):
        """TRPO 更新"""
        # 计算旧策略的分布
        with torch.no_grad():
            _, _, old_dist = self.get_action_and_log_prob(states)
        
        # 计算梯度 g
        _, log_prob, _ = self.get_action_and_log_prob(states)
        loss = -(log_prob * advantages).mean()
        grads = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
        g = torch.cat([grad.view(-1) for grad in grads]).detach()
        
        # 定义 Fisher 向量积 (Fisher-vector product)
        def fisher_vector_product(v):
            kl = self.get_kl_divergence(states, old_dist)
            grads_kl = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.view(-1) for g in grads_kl])
            kl_v = (flat_grad_kl * v).sum()
            grads_kl_v = torch.autograd.grad(kl_v, self.policy.parameters())
            flat_grads_kl_v = torch.cat([g.contiguous().view(-1) for g in grads_kl_v])
            return flat_grads_kl_v + 0.1 * v  # damping
        
        # 共轭梯度求解自然梯度方向
        step_dir = self.conjugate_gradient(fisher_vector_product, g, nsteps=10)
        
        # 线搜索：找到满足 KL 约束的最大步长
        params = list(self.policy.parameters())
        old_params = self.flat_params(params)
        max_step = torch.sqrt(2 * self.delta / (step_dir.dot(fisher_vector_product(step_dir)) + 1e-8))
        full_step = max_step * step_dir
        
        # 线搜索
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            new_params = old_params + alpha * full_step
            self.set_params(params, new_params)
            
            # 检查 KL 约束
            kl = self.get_kl_divergence(states, old_dist).item()
            if kl <= self.delta * 1.5:
                break
        
        # 更新 Critic
        _, log_prob_new, _ = self.get_action_and_log_prob(states)
        return loss.item()

print("TRPO 约束优化 - 核心逻辑实现")
print("注意: TRPO 使用二阶信息（Fisher 矩阵），计算量大于一阶方法")
```

## 深度学习关联

1. **自然梯度在深度学习中的角色**：TRPO 推广了自然梯度法（Natural Gradient）在深度强化学习中的应用。自然梯度考虑了参数空间的 Riemannian 几何结构，在标准监督学习中也有应用（如 Natural Gradient Descent for neural networks），但在高维参数空间中计算 Fisher 矩阵的挑战限制了其广泛应用。
2. **TRPO -> PPO 的演进**：TRPO 虽然是更早的算法，但其理论贡献（策略改进界、信任区域约束）影响深远。PPO 用简单的裁剪近似了 TRPO 的效果，使基于信任区域的方法变得实用化。这种"从理论完善到工程简化"的演进是 DRL 发展的典型模式。
3. **二阶优化与神经网络训练**：TRPO 是少数在实践中成功使用二阶优化（涉及 Hessian/Fisher 矩阵）的深度学习方法之一。大多数深度网络使用一阶 SGD/Adam，因为二阶方法的计算成本过高。TRPO 通过共轭梯度法高效近似求解，以及自然梯度相比于标准梯度的几何优势，使其在策略优化场景中值得额外的计算开销。
