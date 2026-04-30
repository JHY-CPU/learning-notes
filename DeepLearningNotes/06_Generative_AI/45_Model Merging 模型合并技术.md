# 45_Model Merging 模型合并技术

## 核心概念
- **模型合并 (Model Merging)**：将两个或多个具有相同架构的模型的权重进行混合，以在单一模型中组合不同模型的能力。在扩散模型社区中，合并不同的 LoRA 或完整模型来创造新风格。
- **线性插值 (Linear Interpolation / Weight Averaging)**：最简单的合并方式——直接对两个模型的权重做加权平均：$\theta_{\text{merged}} = \alpha \cdot \theta_A + (1 - \alpha) \cdot \theta_B$。
- **SLERP (Spherical Linear Interpolation)**：比线性插值更平滑的合并方法——在权重空间的球面上进行插值，保持权重的范数信息，减少性能损失。
- **TIES-Merging (Trim, Elect Sign, Merge)**：三步合并策略——(1) 修剪细微差异，(2) 通过投票解决符号冲突，(3) 合并一致方向的权重更新。
- **DARE (Drop And REscale)**：通过随机丢弃大部分 delta 权重并重新缩放剩余部分来降低合并中的冲突——简单但有效。
- **模型池 (Model Soup / Model Stock)**：从一个训练轨迹的多个 checkpoints 中平均权重，而不是合并不同任务的模型——常用于提升模型泛化性能。

## 数学推导

**线性合并 (Linear Merge)**：

给定两个模型的权重 $\theta_A$ 和 $\theta_B$（层相同）：

$$
\theta_{\text{merged}} = \alpha \cdot \theta_A + (1 - \alpha) \cdot \theta_B
$$

其中 $\alpha \in [0, 1]$ 控制融合比例。$\alpha=0.5$ 是等权重合并。

**SLERP**：

1. 将权重向量投影到单位球面上：$\hat{\theta}_A = \theta_A / \|\theta_A\|$, $\hat{\theta}_B = \theta_B / \|\theta_B\|$
2. 计算两向量间夹角：$\Omega = \arccos(\hat{\theta}_A \cdot \hat{\theta}_B)$
3. 球面插值：

$$
\theta_{\text{slerp}}(t) = \frac{\sin((1-t)\Omega)}{\sin(\Omega)} \theta_A + \frac{\sin(t\Omega)}{\sin(\Omega)} \theta_B
$$

**TIES-Merging 的数学**：

1. **修剪 (Trim)**：对每个权重参数，如果 $|\theta_A^{(i)} - \theta_B^{(i)}| < \tau$，认为该差异不显著，设为 0
2. **符号投票 (Elect Sign)**：对每个任务 $k$ 的更新 $\Delta_k = \theta_k - \theta_{\text{base}}$，计算整体方向符号 $\gamma^{(i)} = \text{sign}\left(\sum_k \Delta_k^{(i)}\right)$
3. **合并 (Merge)**：只在符号一致的方向上合并：$\theta_{\text{merged}}^{(i)} = \theta_{\text{base}}^{(i)} + \frac{1}{\sum_k \mathbb{1}[\text{sign}(\Delta_k^{(i)}) = \gamma^{(i)}]} \sum_{k: \text{sign}=\gamma} \Delta_k^{(i)}$

**模型插值的 Fisher 加权**：

考虑 Fisher 信息矩阵 $F$（衡量参数的重要性）：

$$
\theta_{\text{merged}} = \frac{F_A \cdot \theta_A + F_B \cdot \theta_B}{F_A + F_B}
$$

Fisher 信息越高，该参数在合并中的权重越大。

## 直观理解
- **模型合并 = 混合两种颜料**：你有红色（擅长赛博朋克风格）和蓝色（擅长水墨风格）两个模型，用 SLERP 混出紫色——得到了一个"赛博朋克水墨风格"模型。
- **为什么不是简单平均**：深度学习模型的权重空间不是欧几里得的——两个优��模型的均值可能是一个"四不像"。SLERP 和 TIES 通过考虑权重的方向和范数来避免这种情况。
- **TIES 的核心洞察**：两个模型可能对某个参数有不同的更新方向——一个想增大，一个想减小。简单的平均会相互抵消（接近 0）。TIES 通过投票决定方向，只沿一致的方向合并。
- **Model Soup = 煮汤时尝不同批次的汤**：训练过程中不同 epoch 的 checkpoint 就像是不同时间尝的汤——每一次的味道有差异。把这些"汤样"平均起来，往往能得到最好的味道（泛化性能）。

## 代码示例

```python
import torch
import numpy as np
import math

class ModelMerging:
    """模型合并工具类"""
    
    @staticmethod
    def linear_merge(model_a_state, model_b_state, alpha=0.5):
        """
        线性插值合并
        
        参数:
            model_a_state: 模型 A 的状态字典
            model_b_state: 模型 B 的状态字典
            alpha: 模型 A 的权重比例 (0-1)
        """
        merged_state = {}
        for key in model_a_state:
            if key in model_b_state:
                merged_state[key] = alpha * model_a_state[key] + \
                                    (1 - alpha) * model_b_state[key]
            else:
                merged_state[key] = model_a_state[key]
        
        return merged_state
    
    @staticmethod
    def slerp_merge(model_a_state, model_b_state, t=0.5):
        """
        SLERP 球面线性插值合并
        
        参数:
            t: 插值参数 (0=完全模型A, 1=完全模型B)
        """
        merged_state = {}
        
        for key in model_a_state:
            if key not in model_b_state:
                merged_state[key] = model_a_state[key]
                continue
            
            v0 = model_a_state[key].flatten().float()
            v1 = model_b_state[key].flatten().float()
            
            # 归一化
            v0_norm = v0 / torch.norm(v0)
            v1_norm = v1 / torch.norm(v1)
            
            # 计算点积（余弦相似度）
            dot = torch.dot(v0_norm, v1_norm)
            dot = torch.clamp(dot, -1.0, 1.0)
            
            # 夹角
            omega = torch.acos(dot)
            
            if omega < 1e-10:
                # 夹角太小，退化为线性插值
                merged = (1 - t) * v0 + t * v1
            else:
                # SLERP
                merged = (torch.sin((1 - t) * omega) / torch.sin(omega)) * v0 + \
                         (torch.sin(t * omega) / torch.sin(omega)) * v1
            
            merged_state[key] = merged.reshape(model_a_state[key].shape)
        
        return merged_state
    
    @staticmethod
    def ties_merge(model_base, model_tasks, trim_threshold=0.01):
        """
        TIES-Merging
        
        参数:
            model_base: 基础模型
            model_tasks: 多个微调模型的列表
            trim_threshold: 修剪阈值
        """
        n_tasks = len(model_tasks)
        delta_tasks = []
        
        # 计算每个任务的 delta
        for task_model in model_tasks:
            delta = {}
            for key in task_model:
                if key in model_base:
                    delta[key] = task_model[key] - model_base[key]
            delta_tasks.append(delta)
        
        merged_state = {}
        for key in model_base:
            # 收集所有任务对该参数的更新
            deltas = torch.stack([dt[key] for dt in delta_tasks], dim=0)
            
            # 1. Trim: 略去小更新
            deltas[torch.abs(deltas) < trim_threshold] = 0
            
            # 2. Elect Sign: 投票决定方向
            sign_sum = torch.sum(torch.sign(deltas), dim=0)
            majority_sign = torch.sign(sign_sum)
            
            # 3. Merge: 只保留多数方向的更新
            mask = (torch.sign(deltas) == majority_sign.unsqueeze(0))
            masked_deltas = deltas * mask.float()
            
            # 取一致方向的均值
            num_agree = mask.float().sum(dim=0)
            num_agree = torch.clamp(num_agree, min=1)
            merged_delta = masked_deltas.sum(dim=0) / num_agree
            
            merged_state[key] = model_base[key] + merged_delta * (torch.abs(majority_sign) > 0).float()
        
        return merged_state

# 演示模型合并
print("=== 模型合并技术 ===")
print()

# 模拟两个模型的权重
model_a = {'layer1.weight': torch.randn(64, 3, 3, 3)}
model_b = {'layer1.weight': torch.randn(64, 3, 3, 3)}

# 线性合并
merged_linear = ModelMerging.linear_merge(model_a, model_b, alpha=0.5)
diff_linear = merged_linear['layer1.weight'] - (0.5 * model_a['layer1.weight'] + 0.5 * model_b['layer1.weight'])
print(f"线性合并差异: {diff_linear.abs().max().item():.6f} (应为 0)")

# SLERP 合并
merged_slerp = ModelMerging.slerp_merge(model_a, model_b, t=0.5)
print(f"SLERP 合并: {merged_slerp['layer1.weight'].shape}")

# TIES 合并（多任务）
model_base = {'conv.weight': torch.randn(64, 3, 3, 3)}
model_task1 = {'conv.weight': model_base['conv.weight'] + torch.randn(64, 3, 3, 3) * 0.1}
model_task2 = {'conv.weight': model_base['conv.weight'] + torch.randn(64, 3, 3, 3) * 0.1}
merged_ties = ModelMerging.ties_merge(model_base, [model_task1, model_task2])

print(f"TIES 合并: {merged_ties['conv.weight'].shape}")

print()
print("合并方法对比:")
print("  线性合并: 简单快速，但可能产生次优结果")
print("  SLERP: 平滑过渡，保留权重范数信息")
print("  TIES: 解决符号冲突，多任务合并首选")
print("  Model Soup: 同任务不同 checkpoint 的均值")
```

## 深度学习关联
- **LoRA 合并**：LoRA 权重可以单独合并（A 矩阵和 B 矩阵分别合并），多个 LoRA 可以合并为一个文件，大大简化了 ComfyUI 等工作流的模型管理。
- **Style Merge (Checkpoint Merging)**：在 Stable Diffusion 社区中，合并不同 checkpoint（如"写实风格"+"动画风格"）是创造新风格模型的常见做法——SD 1.5 生态中有大量"merged checkpoint"。
- **任务向量 (Task Vector)**：定义 $\tau = \theta_{\text{finetuned}} - \theta_{\text{pretrained}}$ 为任务向量。合并不同任务的任务向量可以实现多任务学习——TIES 和 DARE 都基于任务向量框架。
- **模型合并的局限性**：合并不同架构的模型是不可能的，合并相同架构但训练数据差异过大的模型可能产生矛盾（如"一个模型认为猫有 4 条腿，另一个认为有 5 条"）。
