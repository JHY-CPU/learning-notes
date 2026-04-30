# 11_模型压缩：剪枝 (Pruning) 技术与稀疏性

## 核心概念

- **非结构化剪枝 (Unstructured Pruning)**：对模型中的单个权重或神经元进行独立裁剪，将与模型输出重要性较低的权重置为零。不改变网络拓扑结构，但会产生稀疏权重矩阵，需要专门的稀疏计算库（如 cuSPARSE）加速。
- **结构化剪枝 (Structured Pruning)**：按通道、行或整个卷积核等结构化单元进行裁剪，直接减小矩阵尺寸。由于剪枝后权重矩阵保持稠密，可以直接在常规硬件上获得加速，但精度损失通常大于非结构化剪枝。
- **剪枝标准 (Pruning Criterion)**：确定哪些权重被剪掉的依据，常见方法包括：基于权重绝对值大小（Magnitude Pruning）、基于梯度（GraNORM）、基于特征图响应（Activation-based）、基于最终精度影响（Oracle Pruning）。
- **迭代式剪枝 (Iterative Pruning)**：采用"训练-剪枝-微调"循环（Iterative Pruning + Fine-tuning），每次剪掉一定比例（如 10-20%）的权重，然后微调恢复精度。相比一次性大幅剪枝，迭代式剪枝通常能获得更好的最终精度。
- **彩票假设 (Lottery Ticket Hypothesis)**：2019 年 Frankle & Carbin 提出的理论——在一个随机初始化的稠密网络中，存在一个"中奖彩票"子网络，其参数量远小于原网络，但从零训练时具有与原始网络相当的精度。这启发了一种新的剪枝范式。
- **稀疏训练 (Sparse Training)**：在训练过程中就保持权重的稀疏性，而非先稠密训练再剪枝。代表性工作包括 SET (Sparse Evolutionary Training) 和 RigL (Rigged Lottery)，通过动态演化稀疏连接图实现高效训练。

## 数学推导

剪枝可以被形式化为一个约束优化问题：

$$
\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}; \mathcal{D}) \quad \text{s.t.} \quad \|\mathbf{w}\|_0 \leq k
$$

其中 $\|\mathbf{w}\|_0$ 是 $\ell_0$ 范数（非零权重个数），$k$ 是目标非零参数数量。这是一个 NP-hard 的组合优化问题，实践中通常使用贪婪近似。

**基于量级的剪枝 (Magnitude Pruning)** 是最简单的近似策略：

$$
\text{mask}_i = \begin{cases}
0 & \text{if } |w_i| < t \cdot \text{std}(w_{\text{layer}}) \\
1 & \text{otherwise}
\end{cases}
$$

其中 $t$ 是控制剪枝强度的阈值参数。更精确的方法考虑权重的重要性得分：

$$
I(w_i) = |w_i| \cdot \| \nabla_{w_i} \mathcal{L} \|
$$

结合权重大小和梯度信息的综合度量。结构化剪枝中，考虑第 $c$ 个卷积核的重要性：

$$
I(\mathbf{W}_{c,:,:,:}) = \sum_{i,j,k} |W_{c,i,j,k}|
$$

即该卷积核所有权重的 $\ell_1$ 范数之和。范数越小意味着该核的输出对后续层的影响越小。

## 直观理解

- **剪枝 = 修剪果树枝条**：有些枝条不结果或结果很少（权重不重要），剪掉它们可以让营养（计算资源）集中到更有生产力的枝条上（重要权重）。关键是找到"只剪掉多余枝叶但不损伤主干的临界点"。
- **非结构化剪枝 vs 结构化剪枝**：非结构化剪枝像是"剪掉树叶上的小洞"——精度好但加速需要特殊硬件；结构化剪枝像是"剪掉整根枝条"——硬件加速效果好但精度损失更大。
- **最佳实践**：对于追求实际硬件加速的场景（如手机部署），使用结构化剪枝；对于追求极致压缩率的场景（如存储受限的嵌入式设备），使用非结构化剪枝配合稀疏编码。
- **常见陷阱**：剪枝后的模型在训练时的 forward/backward 仍然需要稠密计算（除非使用专门的稀疏框架），所以训练加速有限。真正的加速主要在推理阶段。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

# ========== 1. PyTorch 剪枝 API ==========
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# 1.1 非结构化剪枝：按比例剪枝
prune.l1_unstructured(model[0], name="weight", amount=0.3)
# model[0].weight 被替换为 WeightMask 对象
print(f"剪枝后非零参数比例: {1 - float(torch.mean(model[0].weight_mask)):.2%}")

# 查看可训练参数的变化
for name, param in model[0].named_parameters():
    print(f"  {name}: {param.shape}")

# 1.2 结构化剪枝：按通道剪枝
prune.ln_structured(model[2], name="weight", amount=0.2, n=2, dim=0)
# 沿着 dim=0（输出通道）方向剪枝

# 1.3 自定义剪枝比例（每层不同）
amounts = {model[0]: 0.3, model[2]: 0.2, model[4]: 0.1}
for module, amount in amounts.items():
    prune.l1_unstructured(module, name="weight", amount=amount)

# 1.4 移除剪枝掩码（使剪枝永久化）
for module in [model[0], model[2], model[4]]:
    prune.remove(module, "weight")
# 移除后，weight 不再保留原始未剪枝的副本

# ========== 2. 迭代式剪枝 + 微调 ==========
def iterative_pruning(model, dataloader, target_sparsity=0.8, steps=10):
    """迭代剪枝：每次剪枝后微调"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in model.parameters())

    for step in range(1, steps + 1):
        # 当前目标剪枝比例（线性调度）
        current_sparsity = step / steps * target_sparsity
        for module in model:
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight",
                                      amount=current_sparsity - (step-1)/steps*target_sparsity)

        # 微调几个 epoch
        for epoch in range(3):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        current_params = sum(p.numel() for p in model.parameters() if (p != 0).sum() > 0)
        non_zero = sum((p != 0).sum().item() for p in model.parameters())
        print(f"Step {step}: 剪枝率 {1 - non_zero/total_params:.2%}")

    return model

# ========== 3. 全局剪枝（按所有参数的重要性排序）==========
def global_magnitude_pruning(model, amount=0.5):
    """全局剪枝：将所有参数按绝对值排序，保留 largest"""
    all_weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.append(param.data.abs().view(-1))

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, amount)  # 剪枝阈值

    for name, param in model.named_parameters():
        if "weight" in name:
            mask = param.data.abs() > threshold
            param.data.mul_(mask)
    return model

# ========== 4. 稀疏模型推理加速（需硬件支持）==========
# 对于非结构化稀疏，需要特定的稀疏矩阵乘法库
# PyTorch 的 torch.sparse 模块
def sparse_inference_example():
    dense = torch.randn(1000, 1000).cuda()
    # 制造 90% 稀疏性
    mask = torch.rand_like(dense) > 0.9
    sparse = dense * mask

    # 转为稀疏格式
    sparse_csr = sparse.to_sparse_csr()
    # 使用稀疏矩阵乘法加速
    vec = torch.randn(1000, 64).cuda()
    # result1 = sparse @ vec       # 稠密计算
    # result2 = sparse_csr @ vec   # 稀疏计算（如果支持）

    size_mb_dense = sparse.element_size() * sparse.numel() / 1024**2
    size_mb_sparse = (sparse_csr.col_indices().numel() * 4 +
                      sparse_csr.crow_indices().numel() * 4 +
                      sparse_csr.values().numel() * 4) / 1024**2
    print(f"稠密存储: {size_mb_dense:.2f} MB")
    print(f"稀疏存储: {size_mb_sparse:.2f} MB")
```

## 深度学习关联

- **移动端和边缘设备部署**：剪枝是模型压缩的第一步。在将模型部署到手机、IoT 设备前，通常采用"剪枝 -> 量化 -> 蒸馏"的组合策略。剪枝减少了参数数量，使后续量化更加稳定。
- **剪枝与模型架构搜索 (NAS)**：现代自动化剪枝（如 AMC: AutoML for Model Compression）使用强化学习或贝叶斯优化自动找到每层的最优剪枝率。这在 MLOps 流水线中可以作为一个自动化步骤，每次模型训练后自动搜索最佳剪枝配置。
- **MLOps 中的模型简化流水线**：在生产环境中，剪枝通常集成到模型注册前的 CI 流水线中：注册模型 -> 自动剪枝 -> 评估精度 -> 如果精度损失 < 阈值则发布剪枝版 -> 否则退回微调。这一流程可通过 MLflow 的模型版本管理跟踪。
