# 23_LoRA 在生成模型微调中的应用

## 核心概念

- **LoRA (Low-Rank Adaptation)**：一种高效的模型微调技术，通过在预训练权重矩阵旁添加低秩分解矩阵来适配新任务，冻结原始权重。
- **低秩分解**：用两个小矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 的乘积近似表示增量 $\Delta W = BA$，其中 $r \ll \min(d, k)$，参数量大幅减少。
- **参数量减少**：对 $W \in \mathbb{R}^{d \times k}$，全量微调需要更新 $d \times k$ 个参数；LoRA 只需 $r(d + k)$ 个参数。当 $r=4$ 时，参数量减少到原来的 $1/1000$ 以下。
- **插入式部署**：LoRA 权重独立于基础模型保存和加载，同一个基础模型可以加载不同的 LoRA 权重实现不同风格/角色的切换，无需切换整个模型。
- **缩放因子**：LoRA 的前向计算公式为 $h = W_0x + \frac{\alpha}{r} BAx$，其中 $\alpha$ 是缩放超参数，控制 LoRA 的影响强度。
- **适用层选择**：在扩散模型中，LoRA 通常应用于交叉注意力层的 Q、K、V、O 投影矩阵和文本编码器的部分层。

## 数学推导

**LoRA 的核心数学**：

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将其更新约束为低秩形式：

$$
W_0 + \Delta W = W_0 + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，且 $r \ll \min(d, k)$。

**前向传播**：

$$
h = W_0 x + \Delta W x = W_0 x + BAx = W_0 x + \frac{\alpha}{r} BAx
$$

训练开始时，$A$ 用随机高斯初始化，$B$ 初始化为零，因此 $\Delta W = 0$，微调开始时与原模型行为一致。

**参数量对比**：

以 Stable Diffusion 的交叉注意力层为例：

- QKV 投影权重：$W_q \in \mathbb{R}^{320 \times 768}$（UNet 中）
- 全量微调：$320 \times 768 = 245,760$ 参数
- LoRA ($r=4$)：$A \in \mathbb{R}^{4 \times 768}$ + $B \in \mathbb{R}^{320 \times 4}$ = $3,072 + 1,280 = 4,352$ 参数
- 参数节省：$245,760 / 4,352 \approx 56$ 倍

**与适配器 (Adapter) 的区别**：

- Adapter：在层之间插入新的前馈网络，改变网络深度
- LoRA：在现有权重旁添加并行分支，不改变网络深度
- LoRA 推理时可将 $BA$ 合并到 $W_0$ 中（$W = W_0 + BA$），零推理延迟增加

## 直观理解

- **LoRA = 给已训练好的模型贴一个补丁**：基础模型像是一个通才（什么都会画），LoRA 像一个小补丁（专注于某个风格/角色）。补丁很小，但贴上去后通才就变成了"擅长画动漫角色的通才"。
- **低秩假设**：模型要学习的新知识（比如"赛博朋克风格"）本质上是低维的——不需要改变 10 亿参数就能学会。LoRA 找到的就是这个低维的变化空间。
- **为什么 LoRA 比全量微调好**：全量微调像是给整栋房子重新装修（容易破坏原有的结构），LoRA 像是在墙上挂一幅画（不影响房子主体，也可以随时换画）。
- **r 参数的意义**：$r$ 是 LoRA 的秩，代表新任务需要多少"自由度"。$r=1$ 就像只用黑白两色画画，$r=64$ 就像有 64 种颜色的调色板。一般 $r=4-32$ 就够用了。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """LoRA 层：在权重矩阵旁添加低秩分解"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 冻结原始权重（假设从外部传入）
        self.weight = None  # 原始权重由外部管理
    
    def forward(self, x, original_weight):
        """
        h = Wx + (alpha/r) * BAx
        
        参数:
            x: 输入特征
            original_weight: 原始线性层的权重矩阵
        """
        # 原始输出
        original_out = nn.functional.linear(x, original_weight)
        
        # LoRA 增量
        lora_out = nn.functional.linear(x, self.lora_A)  # x @ A^T
        lora_out = nn.functional.linear(lora_out, self.lora_B)  # (x @ A^T) @ B^T
        
        return original_out + self.scaling * lora_out

class LoRAWrapper(nn.Module):
    """将 LoRA 应用到 Attention 层的包装器"""
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        self.lora = LoRALayer(
            self.in_features, self.out_features, rank, alpha
        )
        
        # 冻结原始权重
        for param in original_linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.lora(x, self.original_linear.weight)

class StableDiffusionWithLoRA(nn.Module):
    """带 LoRA 的 Stable Diffusion（简化版）"""
    def __init__(self, base_model, rank=4, target_modules=['q', 'k', 'v', 'o']):
        super().__init__()
        self.base_model = base_model
        
        # 记录 LoRA 层
        self.lora_layers = nn.ModuleDict()
        
        # 在实际 SD 中，需要遍历 U-Net 的所有交叉注意力层
        # 这里用简化模拟
        for name, module in base_model.named_modules():
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                lora_wrapper = LoRAWrapper(module, rank=rank)
                # 替换原层
                parent = self._get_parent(base_model, name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, lora_wrapper)
                self.lora_layers[name] = lora_wrapper
    
    def _get_parent(self, model, name):
        """获取模块的父模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent
    
    def merge_lora(self):
        """将 LoRA 权重合并到原始权重（推理加速）"""
        for name, lora_layer in self.lora_layers.items():
            # 还原原始 Linear 层
            merged_weight = lora_layer.original_linear.weight.data + \
                (lora_layer.lora.lora_B.data @ lora_layer.lora.lora_A.data) * \
                lora_layer.lora.scaling
            
            new_linear = nn.Linear(
                lora_layer.in_features, lora_layer.out_features,
                bias=lora_layer.original_linear.bias is not None
            )
            new_linear.weight.data = merged_weight
            if lora_layer.original_linear.bias is not None:
                new_linear.bias.data = lora_layer.original_linear.bias.data
            
            # 替换回原始层
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = self._get_parent(self.base_model, parent_name)
            setattr(parent, child_name, new_linear)
    
    def unmerge_lora(self):
        """还原合并的 LoRA 权重"""
        # 需要保存原始权重才能还原（简化实现中略）
        pass

# 训练 LoRA 的典型流程
def train_lora_on_style(model, dataloader, rank=4, lr=1e-4, steps=1000):
    """微调 LoRA 以适配新风格"""
    # 只启用 LoRA 参数的梯度
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    print(f"可训练参数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练比例: {sum(p.numel() for p in trainable_params)/sum(p.numel() for p in model.parameters())*100:.4f}%")
    
    return optimizer

# 演示
print("=== LoRA 微调参数分析 ===")
d, k, r = 320, 768, 4
full_params = d * k
lora_params = r * (d + k)
print(f"全量微调参数: {full_params:,}")
print(f"LoRA 参数 (r={r}): {lora_params:,}")
print(f"参数减少倍数: {full_params / lora_params:.1f}x")
print()
print("LoRA 的关键优势:")
print("1. 单个 LoRA 权重文件仅几 MB")
print("2. 可加载多个 LoRA 并动态切换")
print("3. 推理时可合并到原始权重，零额外延迟")
```

## 深度学习关联

- **LoRA 在 LLM 中的起源**：LoRA 最初是为大语言模型（LLM）微调而设计（Hu et al., 2021），后被社区迁移到扩散模型。在 Stable Diffusion 中，LoRA 主要应用于交叉注意力的线性投影层。
- **LoRA 与 DreamBooth 的组合**：DreamBooth 全量微调整个模型，LoRA 只微调少量参数。实践中常用 LoRA 作为 DreamBooth 的轻量替代，或在 DreamBooth 训练中使用 LoRA 作为底层优化器。
- **LoRA 的变体：DoRA, LoRA-FA**：DoRA（Weight-Decomposed Low-Rank Adaptation）将权重分解为方向和幅度，分别微调；LoRA-FA 只更新 A 矩阵不更新 B 矩阵，进一步减少参数量。
- **多 LoRA 融合 (LoRA Composition)**：多个 LoRA（如一个控制风格、一个控制角色）可以同时加载并叠加效果，为生成式 AI 应用提供了极高的灵活性和可组合性。
