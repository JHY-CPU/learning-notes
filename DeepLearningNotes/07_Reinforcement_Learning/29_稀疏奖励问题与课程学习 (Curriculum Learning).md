# 29_稀疏奖励问题与课程学习 (Curriculum Learning)

## 核心概念

- **稀疏奖励问题 (Sparse Reward Problem)**：在大多数真实场景中，环境只在极少数时刻给出非零奖励（如机器人成功抓住杯子时才得 1 分），大多数时间奖励为 0。这导致智能体几乎没有信号来学习，无法区分不同动作的好坏。
- **奖励塑造 (Reward Shaping)**：人工设计辅助奖励函数来引导学习（如机械臂向目标移动时给予小奖励）。但需小心——不恰当的奖励塑造会诱导智能体"钻漏洞"（reward hacking）。
- **内在动机 (Intrinsic Motivation)**：当外部奖励稀疏时，智能体产生"内在奖励"来驱动探索。常见方法包括基于新奇度（novelty）、预测误差（prediction error）和信息增益（information gain）的内在奖励。
- **课程学习 (Curriculum Learning)**：将训练过程组织为从易到难的课程，让智能体逐步掌握复杂技能。最早由 Bengio (2009) 在监督学习中提出，在 RL 中特别有效。
- **自动课程学习 (Automatic Curriculum Learning)**：不依赖手工设计课程，而是让算法自动选择"处于智能体能力边界上"的任务。如通过学生网络的能力水平动态调整任务难度分布。
- **逆向课程生成 (Reverse Curriculum Generation)**：从目标状态出发，反向生成逐渐远离目标的训练状态。先学习离目标近的状态，再逐步向远处扩展，有效解决了从初始状态到目标状态"一步跨越太大"的问题。

## 数学推导

$$
\text{标准 RL 目标: } \max_\pi \mathbb{E}_\pi \left[ \sum_{t} \gamma^t r_t^{\text{ext}} \right]
$$

$$
\text{内在动机: } r_t^{\text{total}} = r_t^{\text{ext}} + \beta \cdot r_t^{\text{int}}
$$

$$
\text{新奇度探索 (ICM): } r_t^{\text{int}} = \|\phi(s_{t+1}) - \hat{\phi}(s_{t+1})\|^2
$$

$$
\text{其中 } \phi(\cdot) \text{ 是状态特征编码器，} \hat{\phi} \text{ 是前向动力学预测}
$$

$$
\text{预测误差探索 (RND): } r_t^{\text{int}} = \|f(s_{t+1}) - \hat{f}(s_{t+1})\|^2
$$

$$
\text{其中 } f \text{ 是固定随机目标网络，} \hat{f} \text{ 是训练中的预测器}
$$

$$
\text{Hindsight Experience Replay (HER): }
$$

$$
\text{原始轨迹: } (s_t, a_t, r_t, s_{t+1}, g) \quad \text{其中 } g \text{ 是原始目标}
$$

$$
\text{修改后: } (s_t, a_t, r_t', s_{t+1}, g') \quad \text{其中 } g' = s_{t+1} \text{（达成目标）}
$$

**推导说明**：
- ICM (Intrinsic Curiosity Module) 的内在奖励基于"状态预测的难度"：模型难以预测下一步状态的区域通常具有更高的探索价值。
- RND (Random Network Distillation) 的内在奖励基于"预测误差"：固定随机网络输出与训练网络输出之间的差异，代表该状态的新奇程度。
- HER 的核心洞察：即使智能体没有达到原始目标 $g$，它实际上达到了某个其他状态 $s_{t+1}$。将这个"实际达到的状态"视为新目标，这条轨迹就变成了"成功轨迹"。这相当于自动生成奖励信号。

## 直观理解

稀疏奖励和课程学习可以类比为"教小孩学骑自行车"：

**稀疏奖励问题**：你给小孩一辆自行车，说"骑到街角给你 100 元"。小孩上车就摔——0 元。再摔——0 元。摔了 100 次，一分钱没拿到，完全不知道自己在干嘛。这就是稀疏奖励：只在最终成功时给奖励，中间毫无反馈。

**常见的解决方案**：

1. **奖励塑形（手把手教）**：你说"你蹬了一下脚踏板，奖励 1 元！你保持了 2 秒平衡，再奖 1 元！"——这可以，但你得一直在旁边盯着，而且可能诱导小孩为了拿奖励而"作弊"（比如一直蹬不拐弯）。

2. **内在动机（好奇宝宝）**：小孩自己觉得"上次我只能保持 3 秒平衡，这次我要试试保持 5 秒！"——这种"挑战自我"的内在驱动力就是内在动机。即使外面不给奖励，满足好奇心本身就是奖励。

3. **课程学习（循序渐进）**：
   - 第一天：用辅助轮学骑行（最容易）
   - 第二天：拆掉辅助轮，在草地学（摔了不疼）
   - 第三天：在平路上学（常规难度）
   - 第四天：学拐弯（更难）
   这就是课程学习——从易到难，每一步都是前一步的自然延伸。

4. **自动课程学习（自适应难度）**：
   聪明的教练会观察小孩的进步速度，动态调整难度。"你现在稳了，我们来试试上坡"——这比固定课程更高效。

## 代码示例

```python
import numpy as np
import gymnasium as gym
from collections import deque
import random

# --- Hindsight Experience Replay (HER) 示例 ---
class HERBuffer:
    """支持 HER 的经验回放缓冲区"""
    def __init__(self, capacity=100000, her_ratio=0.8):
        self.buffer = deque(maxlen=capacity)
        self.her_ratio = her_ratio
    
    def add_episode(self, episode):
        """添加完整 episode 并生成 HER 样本"""
        # episode: list of (s, a, r, s', done, goal)
        states, actions, rewards, next_states, dones, goals = zip(*episode)
        
        for t in range(len(states)):
            # 添加原始 transition
            self.buffer.append((
                states[t], actions[t], rewards[t], 
                next_states[t], dones[t], goals[t]
            ))
        
        # HER: 将"实际达到的状态"作为新目标
        actual_goal = next_states[-1]  # episode 结束时的状态
        for t in range(len(states)):
            if np.random.random() < self.her_ratio:
                # 计算新奖励：是否达到了实际目标
                new_reward = 1.0 if np.linalg.norm(
                    next_states[t] - actual_goal) < 0.1 else 0.0
                self.buffer.append((
                    states[t], actions[t], new_reward,
                    next_states[t], dones[t], actual_goal
                ))

# --- ICM (Intrinsic Curiosity Module) 示例 ---
class ICM:
    """内在好奇心模块"""
    def __init__(self, state_dim, action_dim, hidden=256):
        # 状态编码器
        self.encoder = lambda s: s  # 简化: 使用原始状态
        # 前向动力学模型: 预测 next_state 的编码
        self.forward_model = lambda s, a: np.random.randn(state_dim)  # 简化
        
    def intrinsic_reward(self, state, action, next_state):
        # 计算预测误差作为内在奖励
        predicted_next = self.forward_model(state, action)
        actual_next = self.encoder(next_state)
        return np.linalg.norm(predicted_next - actual_next)

# --- 课程学习示例 ---
class CurriculumScheduler:
    """课程学习调度器"""
    def __init__(self, difficulty_levels, success_threshold=0.8):
        self.levels = sorted(difficulty_levels)
        self.current_level = 0
        self.success_threshold = success_threshold
        self.recent_successes = deque(maxlen=100)
    
    def get_current_task(self):
        """返回当前难度级别的任务参数"""
        return self.levels[self.current_level]
    
    def record_outcome(self, success):
        """记录本轮是否成功，判断是否提升难度"""
        self.recent_successes.append(success)
        if len(self.recent_successes) >= 50:
            success_rate = np.mean(self.recent_successes)
            if success_rate > self.success_threshold:
                # 提升难度
                self.current_level = min(
                    self.current_level + 1, 
                    len(self.levels) - 1)
                print(f"课程升级! 当前难度: {self.current_level}")
    
    def is_complete(self):
        return self.current_level == len(self.levels) - 1

# --- 自动课程学习 (GoalGAN) 简单示意 ---
class GoalGAN:
    """自动生成处于能力边界的目标"""
    def __init__(self, state_dim):
        # 实际应使用 GAN 架构生成目标
        # 这里简化为难度插值
        pass
    
    def generate_goal(self, current_success_rate):
        """根据当前成功率生成合适难度的目标"""
        if current_success_rate > 0.8:
            # 成功率太高 -> 生成更难的目标
            return "harder_goal"
        elif current_success_rate < 0.2:
            # 成功率太低 -> 生成更易的目标
            return "easier_goal"
        else:
            return "current_level_goal"

print("稀疏奖励问题与课程学习 - 核心概念整理")
print("""
解决方法对比:
  奖励塑形     - 快速但可能诱导作弊行为
  内在动机     - 通用但计算开销较大
  HER          - 自动生成奖励信号，特别适合目标导向任务
  课程学习     - 从易到难，效果好但需设计课程
  自动课程     - 自适应难度调整，最灵活但实现复杂
""")
```

## 深度学习关联

1. **课程学习在 LLM 训练中的应用**：大语言模型的训练过程本身就是课程学习——从简单的"预测下一个词"开始，到复杂的指令遵循、推理任务。RLHF 中的 PPO 训练也经常使用课程：先训练简单 prompt，逐步提升到复杂 prompt。
2. **好奇心驱动探索与自监督学习**：ICM 和 RND 的内在奖励机制与自监督学习（self-supervised learning）高度相关。两者都基于"预测"任务——如果模型能够准确预测，说明该数据点已经被理解（不需要探索）；如果预测误差大，说明该数据点有新的信息值得探索。这与主动学习（active learning）中的不确定性采样的思路一致。
3. **HER 与目标条件 RL**：HER 不仅解决了稀疏奖励问题，还推动了"目标条件 RL"（Goal-conditioned RL）的发展。在目标条件 RL 中，价值函数和策略都以目标 $g$ 为条件 $V(s, g)$，HER 自动为每个 trajectory 提供了多个"伪目标"，在机器人和操作任务中取得了显著成功。
