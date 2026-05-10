# Paged Attention 与长序列方案


# Paged Attention 与长序列方案


#### 核心问题
LLM 推理服务中，KV Cache 的内存管理是一个关键挑战：不同请求的序列长度不同且动态增长，传统的连续内存分配方式导致严重的内存碎片（内部碎片和外部碎片）。Paged Attention 借鉴操作系统中虚拟内存的分页思想，解决了这一问题。


## 1. KV Cache 内存管理问题


### 1.1 传统方案的浪费


传统 KV Cache 实现为每个请求预分配连续内存，最大长度为 max_seq_len：


```
# 传统方案: 为每个请求预分配最大长度的连续内存
# 请求 A: 实际生成 100 tokens，但预分配 2048 tokens 的空间
# 请求 B: 实际生成 500 tokens，但预分配 2048 tokens 的空间
# 浪费: 60-90% 的预分配空间未被使用

# 内存碎片类型:
# 1. 内部碎片: 已分配但未使用的空间（预分配 > 实际使用）
# 2. 外部碎片: 空闲空间分散，无法满足新请求的大块连续需求
# 总利用率通常只有 30-50%
```


### 1.2 内存浪费的来源


| 浪费来源 | 描述 | 典型浪费比例 |
| --- | --- | --- |
| 预留浪费 | 按最大长度预分配，实际只用一部分 | 40-60% |
| 碎片浪费 | 请求结束后留下碎片空间，无法被新请求利用 | 20-30% |
| 并行浪费 | batch 中长短不一，短序列等待长序列完成时的空间浪费 | 10-20% |


## 2. PagedAttention（vLLM）


### 2.1 核心思想


PagedAttention（Kwon et al., 2023, vLLM）借鉴操作系统中**虚拟内存分页**的概念：将 KV Cache 分成固定大小的 block（页），通过一个 block table（页表）映射逻辑位置到物理位置。不需要连续的物理内存，只需要逻辑上连续。


### 2.2 Block Table（块表）


$$
逻辑地址: token_position → block_table[position / block_size][position % block_size]
        block_table: 请求的逻辑页号 → 物理页号的映射表
        block: 固定大小的 KV Cache 存储单元（如 16 个 token 的 KV）
$$


```
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim):
        """
        num_blocks: 总物理块数（由总显存决定）
        block_size: 每块容纳的 token 数（如 16）
        """
        self.block_size = block_size
        # 物理 KV 缓存池: (num_blocks, num_layers, 2, num_heads, block_size, head_dim)
        self.blocks = torch.zeros(num_blocks, num_layers, 2, num_heads, block_size, head_dim)
        # 空闲块列表
        self.free_blocks = list(range(num_blocks))

    def allocate(self, seq_len):
        """为新请求分配逻辑块"""
        num_needed = (seq_len + self.block_size - 1) // self.block_size
        allocated = []
        for _ in range(num_needed):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available")
            allocated.append(self.free_blocks.pop(0))
        return allocated  # 返回物理块号列表

    def append_token(self, block_table, position, k, v):
        """追加一个 token 的 KV 到指定逻辑位置"""
        block_idx = position // self.block_size
        block_offset = position % self.block_size

        # 需要新块？
        if block_idx >= len(block_table):
            block_table.append(self.allocate(1)[0])

        phys_block = block_table[block_idx]
        self.blocks[phys_block, :, 0, :, block_offset, :] = k  # K
        self.blocks[phys_block, :, 1, :, block_offset, :] = v  # V
```


### 2.3 注意力计算中的块访问


```
def paged_attention(Q, block_table, kv_cache, block_size):
    """
    Q: (num_heads, head_dim) - 当前 token 的查询
    block_table: list[int] - 逻辑→物理块映射
    kv_cache: 物理块池
    """
    output = torch.zeros_like(Q)
    max_score = float('-inf')
    sum_exp = 0.0

    for logical_idx, phys_block in enumerate(block_table):
        # 加载该物理块的 K, V
        K_block = kv_cache[phys_block, :, 0]  # (num_heads, block_size, head_dim)
        V_block = kv_cache[phys_block, :, 1]

        # 计算该块的注意力分数
        scores = (Q.unsqueeze(1) * K_block).sum(-1) / sqrt(head_dim)

        # 在线 softmax 更新
        block_max = scores.max()
        new_max = max(max_score, block_max)

        # 修正之前的累积值
        correction = math.exp(max_score - new_max) if max_score > -float('inf') else 1.0
        output = output * correction
        sum_exp = sum_exp * correction

        # 累积当前块
        exp_scores = torch.exp(scores - new_max)
        sum_exp += exp_scores.sum()
        output += (exp_scores.unsqueeze(-1) * V_block).sum(1)

        max_score = new_max

    return output / sum_exp
```


### 2.4 PagedAttention 的优势


| 特性 | 传统方案 | PagedAttention |
| --- | --- | --- |
| 内存分配 | 连续预分配 | 按需分块分配 |
| 内存利用率 | 30-50% | 接近 100% |
| 碎片类型 | 内外碎片均严重 | 仅最后一页有少量内部碎片 |
| 动态长度支持 | 差（受预分配限制） | 天然支持 |
| 并行采样 | 每个分支独立分配 | 分支间共享前缀块（copy-on-write） |
| 实现复杂度 | 低 | 中等（需管理 block_table） |


## 3. Ring Attention


### 3.1 动机


当序列长度超过单 GPU 显存容量时，需要将序列分布在多个 GPU 上。Ring Attention（Liu et al., 2023）提出了一种高效的分布式注意力计算方案，将 KV 块以环形拓扑在 GPU 间传递，实现任意长序列的注意力计算。


### 3.2 算法原理


```
# Ring Attention 的计算过程
# 假设 N 个 GPU，序列分为 N 块：[Q0,K0,V0], [Q1,K1,V1], ..., [QN-1,KN-1,VN-1]
# 每个 GPU i 持有 Qi, Ki, Vi

# 初始化: GPU i 加载自己的 Qi, Ki, Vi
# 迭代 N 步（以 GPU 0 为例）:
#   步骤 1: GPU0 计算 Attention(Q0, K0, V0)，同时发送 K0,V0 → GPU1
#   步骤 2: GPU0 接收 K_{N-1}, V_{N-1}，计算 Attention(Q0, K_{N-1}, V_{N-1})
#           同时发送 K_{N-1},V_{N-1} → GPU1
#   ...
#   步骤 N: GPU0 接收 K1, V1，计算 Attention(Q0, K1, V1)

# 每个 GPU 在 N 步中累积完整的注意力结果
# KV 块在 GPU 间形成"环形"流动
```


#### Ring Attention 的关键特性


- **线性显存增长**
   ：每个 GPU 的显存只需容纳序列的 1/N，总序列长度可线性扩展到 N × 单 GPU 容量
- **计算与通信重叠**
   ：每个 GPU 在等待下一块 KV 传输时，可以计算当前块的注意力
- **与 Blockwise Parallel Transformer 结合**
   ：将注意力和前馈网络的计算融合，进一步优化
- **支持因果掩码**
   ：自回归场景下，GPU 只需要等待它关注的 KV 块到达即可开始计算


## 4. Sequence Parallelism（序列并行）


### 4.1 基本思想


序列并行将长序列沿序列维度切分到多个设备上，每个设备只持有序列的一个片段。这与 Ring Attention 密切相关，但更广泛地涵盖了注意力、前馈网络和层归一化的分布式计算。


### 4.2 实现方案对比


| 方案 | 切分维度 | 通信模式 | 适用场景 |
| --- | --- | --- | --- |
| Tensor Parallelism | 模型维度 | All-Reduce | 模型太大放不下 |
| Pipeline Parallelism | 层维度 | P2P 发送 | 模型太深 |
| Sequence Parallelism | 序列维度 | All-Gather / Reduce-Scatter | 序列太长 |
| Ring Attention | 序列维度（KV） | Send / Recv 环形 | 超长序列注意力 |


```
# 序列并行的基本框架
# 假设 seq_len = 16384, world_size = 4
# 每个 GPU 处理 16384/4 = 4096 个 token

# 层归一化: 直接在本地计算（只依赖当前 token）
# 前馈网络: 直接在本地计算（只依赖当前 token）
# 注意力: 需要跨设备获取其他片段的 KV

# 方案1: All-Gather 所有 KV 后本地计算（简单但通信量大）
# all_gather(local_KVs) → 每个 GPU 拿到完整 KV → 本地计算注意力

# 方案2: Ring Attention（通信高效）
# KV 块在 GPU 间环形传递，计算与通信重叠

# 方案3: Ulysses（DeepSpeed）
# 通过 All-to-All 通信在注意力头维度和序列维度间切换
```


## 5. 长上下文方案对比


### 5.1 各方案概览


| 方案 | 核心思路 | 序列长度上限 | 质量 | 代表模型/框架 |
| --- | --- | --- | --- | --- |
| Full Attention | 全局注意力 | 受显存限制（~8K） | 最优 | 原始 Transformer |
| Flash Attention | IO 优化 | ~64K（单卡） | 最优（精确） | 几乎所有现代框架 |
| Sliding Window | 局部窗口 | 无上限 | 长距离略降 | Mistral, Gemma |
| Ring Attention | 分布式序列并行 | 理论无上限 | 最优 | 多种开源框架 |
| Longformer/BIGBIRD | 稀疏注意力 | 16K-128K | 良好 | Longformer |
| Mamba/SSM | 状态空间模型 | 理论无上限 | 接近 Transformer | Mamba, RWKV |
| Retrieval Augmented | 检索增强 | 无上限 | 取决于检索质量 | RAG 系统 |


### 5.2 实际工程选择


> **Warning:** #### 当前（2024-2025）的工程实践
>
>
> - **4K-32K 序列**
>    ：Flash Attention 2 + GQA + RoPE（最常见场景，如代码补全、对话）
> - **32K-128K 序列**
>    ：Flash Attention 2 + 序列并行 / Ring Attention + 位置编码外推（如文档处理）
> - **128K+ 序列**
>    ：Ring Attention 多机并行 + YaRN 外推 + 可能的稀疏注意力混合（如长文档总结）
> - **超长上下文**
>    ：考虑 RAG 方案替代暴力注意力（检索相关段落后拼接）


## 6. 实际部署中的完整优化栈


```
# 现代 LLM 推理系统的完整优化栈
# 以 vLLM 为例:

# 1. 模型层优化
#    - GQA 减少 KV Cache 大小
#    - RoPE 位置编码

# 2. 计算层优化
#    - Flash Attention 2 (IO-aware)
#    - FP8/INT8 量化 (Flash Attention 3)

# 3. 内存层优化
#    - PagedAttention (消除碎片)
#    - KV Cache 量化 (INT8/INT4)

# 4. 调度层优化
#    - Continuous Batching（动态批处理）
#    - 抢占式调度（Preemption）

# 5. 并行层优化
#    - Tensor Parallelism（模型并行）
#    - Pipeline Parallelism（流水线并行）
#    - Sequence Parallelism / Ring Attention（长序列）
```


## 7. 总结


#### 核心要点


1. **PagedAttention**
   是推理服务的关键创新，通过分页管理将 KV Cache 内存利用率从 30-50% 提升到接近 100%
2. **Ring Attention**
   通过环形 KV 传递实现跨 GPU 的超长序列注意力，计算与通信重叠
3. **序列并行**
   是处理超长序列的基本范式，可与 Ring Attention、Ulysses 等结合
4. 实际部署中多种技术
   **组合使用**
   ：Flash Attention + PagedAttention + GQA + 序列并行
5. 长上下文方案选择取决于
   **序列长度、硬件条件和质量要求**
   的权衡

PagedAttention与长序列方案 - 分页缓存、Ring Attention、序列并行、长上下文方案对比完整笔记


<!-- Converted from: 03_Paged_Attention与长序列.html -->
