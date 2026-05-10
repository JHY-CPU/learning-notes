# KV Cache 与推理优化


# KV Cache 与推理优化


#### 核心问题


在自回归推理中，生成每个新 token 都需要计算注意力，而前面 token 的 K 和 V 不会改变。如果不做缓存，每次生成都要重新计算所有历史 token 的 KV，导致大量重复计算。KV Cache 通过缓存已计算的 KV 向量来消除这种冗余，是 LLM 推理的基础优化手段。


## 1. KV Cache 原理


### 1.1 问题背景


自回归语言模型逐 token 生成：每一步都依赖于之前所有 token 的上下文。以生成第 t 个 token 为例：


- 需要计算注意力：q
   ~t~
   对 k
   ~1~
   , k
   ~2~
   , ..., k
   ~t~
   的注意力权重
- 如果没有缓存，需要重新计算 k
   ~1:t-1~
   和 v
   ~1:t-1~
   （虽然结果与之前完全相同）
- 缓存后，只需计算当前 token 的 q
   ~t~
   , k
   ~t~
   , v
   ~t~
   ，然后与缓存的 k
   ~1:t-1~
   , v
   ~1:t-1~
   拼接


### 1.2 缓存结构


KV Cache 存储每个注意力层中每个 token 的 Key 和 Value 向量：


```
# KV Cache 数据结构
# cache shape: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
# 其中 2 代表 K 和 V 两个矩阵

class KVCache:
    def __init__(self):
        self.key_cache = []    # list of (batch, heads, seq_len, head_dim)
        self.value_cache = []  # list of (batch, heads, seq_len, head_dim)

    def update(self, layer_idx, new_key, new_value):
        """追加新的 KV 到缓存"""
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(new_key)
            self.value_cache.append(new_value)
        else:
            # 沿 seq_len 维度拼接
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], new_key], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], new_value], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```


### 1.3 带 KV Cache 的注意力计算


```
def attention_with_cache(Q, K_new, V_new, cache, layer_idx):
    """
    Q: 当前 token 的查询向量 (batch, heads, 1, head_dim)
    K_new, V_new: 当前 token 的键值 (batch, heads, 1, head_dim)
    cache: KVCache 对象
    """
    # 更新缓存，获取完整的 K, V
    K, V = cache.update(layer_idx, K_new, V_new)
    # K, V: (batch, heads, total_seq_len, head_dim)

    # 标准注意力计算
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output
```


## 2. Prefill 与 Decode 两阶段


### 2.1 Prefill 阶段（提示词处理）


Prefill 阶段处理整个输入 prompt，一次性计算所有 token 的 KV 并缓存：


| 特性 | Prefill |
| --- | --- |
| 输入 | 完整 prompt（n 个 token） |
| 计算模式 | 并行计算所有 token（类似训练） |
| 计算复杂度 | O(n²·d)（全注意力） |
| 瓶颈 | 计算密集（compute-bound） |
| KV Cache 状态 | 初始化，填入 n 个 token 的 KV |
| 输出 | 第一个生成 token 的 logits + 完整 KV Cache |


### 2.2 Decode 阶段（逐 token 生成）


Decode 阶段每次只生成一个 token，利用 KV Cache 避免重复计算：


| 特性 | Decode |
| --- | --- |
| 输入 | 单个新 token |
| 计算模式 | 逐 token 自回归生成 |
| 计算复杂度 | O(n·d)（只需计算当前 token） |
| 瓶颈 | 内存带宽（memory-bound） |
| KV Cache 状态 | 每次追加 1 个 token 的 KV |
| 输出 | 下一个 token 的 logits |


```
def generate_with_kv_cache(model, prompt_tokens, max_new_tokens):
    """完整的 KV Cache 推理流程"""
    cache = KVCache()

    # ===== Prefill 阶段 =====
    # 处理整个 prompt
    logits = model(prompt_tokens, cache=cache, is_prefill=True)
    next_token = sample(logits[:, -1, :])  # 取最后一个位置的 logits

    generated = [next_token]

    # ===== Decode 阶段 =====
    for step in range(max_new_tokens - 1):
        # 只输入最新生成的 1 个 token
        logits = model(next_token, cache=cache, is_prefill=False)
        next_token = sample(logits[:, -1, :])
        generated.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return generated
```


## 3. KV Cache 显存计算


### 3.1 计算公式


KV Cache 的显存占用是推理时的主要内存开销之一：


$$
KV Cache 内存 = 2 × num_layers × batch_size × num_heads × seq_len × head_dim × dtype_bytes
$$


其中 2 代表 K 和 V 两个矩阵。也可以等价写为：


$$
KV Cache 内存 = 2 × L × B × S × dmodel × dtype_bytes   （标准 MHA）
        KV Cache 内存 = 2 × L × B × S × nkv_heads × dhead × dtype_bytes   （GQA/MQA）
$$


### 3.2 实际显存估算


| 模型 | Layers | Heads (KV) | Head Dim | 4K 序列显存 | 32K 序列显存 |
| --- | --- | --- | --- | --- | --- |
| LLaMA-7B (MHA) | 32 | 32 | 128 | 4 GB | 32 GB |
| LLaMA-7B (GQA, 8 KV heads) | 32 | 8 | 128 | 1 GB | 8 GB |
| Mixtral-8x7B | 32 | 8 | 128 | 1 GB | 8 GB |
| GPT-3 175B | 96 | 96 | 128 | 12 GB | 96 GB |


> **Warning:** #### 显存计算示例（LLaMA-7B, GQA）
>
>
> ```
> # LLaMA-7B: 32 layers, 8 KV heads (GQA), head_dim=128, fp16
> # seq_len = 4096
> kv_cache_size = 2 * 32 * 1 * 8 * 4096 * 128 * 2  # bytes
>              = 2 * 32 * 8 * 4096 * 128 * 2
>              = 4,294,967,296 bytes
>              ≈ 4 GB (fp16) 或 ≈ 2 GB (int8 量化)
> ```


### 3.3 影响显存的关键因素


- **序列长度**
   ：KV Cache 与序列长度成线性关系，是最大的影响因子
- **批大小**
   ：每个请求都需要独立的 KV Cache，批大小直接倍增开销
- **KV 头数**
   ：使用 GQA/MQA 可显著减少 KV Cache（见下文）
- **数据类型**
   ：KV Cache 量化到 int8 或 int4 可减少 2-4 倍显存
- **层数**
   ：更深的模型需要更多层的缓存


## 4. Grouped Query Attention (GQA)


### 4.1 从 MHA 到 MQA 到 GQA


GQA 是 MHA 和 MQA 之间的折中方案，在保持模型质量的同时大幅减少 KV Cache 大小：


| 方案 | Q 头数 | KV 头数 | KV Cache 减少 | 质量 |
| --- | --- | --- | --- | --- |
| **MHA**（Multi-Head） | h | h | 1x（基准） | 最佳 |
| **GQA**（Grouped） | h | g (g < h) | h/g 倍 | 接近 MHA |
| **MQA**（Multi-Query） | h | 1 | h 倍 | 略有下降 |


$$
GQA: 将 h 个 Q 头分成 g 组，每组共享同一对 KV 头
        每组包含 h/g 个 Q 头，共享 1 个 K 头和 1 个 V �头
$$


### 4.2 GQA 实现细节


```
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.group_size = num_q_heads // num_kv_heads  # 每组几个 Q 头共享 1 个 KV 头

        self.W_Q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        K = self.W_K(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        V = self.W_V(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch, q_heads, seq, head_dim)
        K = K.transpose(1, 2)  # (batch, kv_heads, seq, head_dim)
        V = V.transpose(1, 2)

        # 关键: 将 KV 头扩展以匹配 Q 头数
        # 每 group_size 个 Q 头共享 1 个 KV 头
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, q_heads, seq, head_dim)
        V = V.repeat_interleave(self.group_size, dim=1)

        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.W_O(output)
```


### 4.3 GQA 的实际应用


| 模型 | Q 头数 | KV 头数 | 分组比 | KV Cache 节省 |
| --- | --- | --- | --- | --- |
| LLaMA-2 70B | 64 | 8 | 8:1 | 8x |
| LLaMA-3 8B | 32 | 8 | 4:1 | 4x |
| Mistral 7B | 32 | 8 | 4:1 | 4x |
| Gemma 7B | 16 | 16 | 1:1 (MHA) | 1x |
| Falcon 40B | 64 | 1 | 64:1 (MQA) | 64x |


## 5. Multi-Query Attention (MQA)


### 5.1 原理


MQA 是 GQA 的极端情况：所有 Q 头共享唯一的一对 KV 头（g=1）。这可以最大化减少 KV Cache，但可能导致一定的质量损失。


$$
MQA: num_kv_heads = 1
        所有 Q 头使用相同的 K 和 V，KV Cache 减少到 1/h
$$


#### MQA 的优缺点


- **优点**
   ：KV Cache 减少 h 倍，推理速度更快（更少的内存带宽需求）
- **优点**
   ：模型参数量减少（KV 投影矩阵更小）
- **缺点**
   ：所有 Q 头共享同一 KV 表示，可能降低模型表达能力
- **实践结论**
   ：GQA（如 8 KV 头）在质量和效率间取得了更好的平衡，已成为主流选择


## 6. KV Cache 量化


### 6.1 量化策略


KV Cache 量化将缓存的 KV 值从 float16/bfloat16 压缩到更低的精度，进一步减少显存占用：


| 量化方案 | 精度 | 显存减少 | 质量影响 | 代表工作 |
| --- | --- | --- | --- | --- |
| FP16（基准） | 16 bit | 1x | 无 | - |
| INT8 量化 | 8 bit | 2x | 极小 | SmoothQuant |
| INT4 量化 | 4 bit | 4x | 轻微 | KIVI, QQQ |
| Mixed 精度 | 4/8/16 bit 混合 | 2-4x | 极小 | KVQuant |


### 6.2 逐通道 vs 逐 token 量化


```
# KV Cache 量化的两种粒度

# 1. 逐 token 量化 (per-token)
# 每个 token 的 KV 向量有独立的 scale/zero_point
# 更精确，但额外开销更大
# scale shape: (batch, heads, seq_len, 1)

# 2. 逐通道量化 (per-channel)
# 每个 head 的每个通道有独立的量化参数
# 更紧凑，精度略低
# scale shape: (batch, heads, 1, head_dim)

# 3. 分组量化 (group-wise)
# 每 group_size 个元素共享量化参数
# 平衡精度和开销
# 如 group_size=128: 每 128 个元素一组
```


> **Warning:** #### K 量化 vs V 量化的差异
>
>
> 研究表明，K（Key）对量化更敏感，因为 Key 的分布直接影响注意力分数的计算。V（Value）对量化容忍度更高。因此一些方法对 K 使用更高精度（如 K 用 int8，V 用 int4）来平衡质量和效率。


## 7. 推理优化总结


### 7.1 优化层次总结


| 优化层次 | 技术 | 效果 |
| --- | --- | --- |
| 计算优化 | KV Cache | 避免重复计算，decode 速度提升 10-50x |
| 架构优化 | GQA / MQA | KV Cache 减少 4-64x |
| 精度优化 | KV Cache 量化 | KV Cache 再减少 2-4x |
| 内存优化 | PagedAttention | 消除 KV Cache 内存碎片 |
| 调度优化 | Continuous Batching | 提高 GPU 利用率 |
| IO 优化 | FlashAttention | 减少 HBM 访问，加速注意力计算 |


### 7.2 KV Cache 显存管理关键公式速查


$$
单个 token KV 占用 = 2 × L × nkv_heads × dhead × dtype_bytes
        KV Cache 总量 = 单 token 占用 × batch_size × seq_len
        GQA 节省比 = num_q_heads / num_kv_heads
$$

KV Cache与推理优化 - 缓存原理、Prefill/Decode阶段、显存计算、GQA/MQA、KV Cache量化完整笔记


<!-- Converted from: 03_KV_Cache与推理优化.html -->
