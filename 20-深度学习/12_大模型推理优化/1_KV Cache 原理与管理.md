# 1_KV Cache 原理与管理

## 1. KV Cache 基本原理

在自回归生成中，每生成一个新 token 都需要计算注意力。如果不做优化，每个 token 都要**重新计算所有历史 token 的 Key 和 Value**，导致大量重复计算。

```
无 KV Cache:
  Step 1: 计算 K₁, V₁                        → O(d)
  Step 2: 重新计算 K₁,V₁ + K₂,V₂              → O(2d)
  Step 3: 重新计算 K₁,V₁ + K₂,V₂ + K₃,V₃     → O(3d)
  总计: O(1+2+3+...+n) = O(n²) 计算量

有 KV Cache:
  Step 1: 计算 K₁, V₁, 缓存                   → O(d)
  Step 2: 只计算 K₂, V₂, 读缓存 K₁,V₁          → O(d)
  Step 3: 只计算 K₃, V₃, 读缓存                → O(d)
  总计: O(n) 计算量（每个 step 一次）
```

### 注意力计算中的 KV Cache

```python
def attention_with_kv_cache(Q_new, K_cache, V_cache, W_Q, W_K, W_V):
    """
    Q_new: 当前 token 的查询向量 [1, d]
    K_cache: 历史所有 token 的 Key 缓存 [seq_len, d]
    V_cache: 历史所有 token 的 Value 缓存 [seq_len, d]
    """
    # 1. 计算当前 token 的 K, V
    K_new = Q_new @ W_K  # [1, d]
    V_new = Q_new @ W_V  # [1, d]

    # 2. 追加到缓存
    K_cache = concatenate([K_cache, K_new])  # [seq_len+1, d]
    V_cache = concatenate([V_cache, V_new])  # [seq_len+1, d]

    # 3. 计算注意力
    scores = Q_new @ K_cache.T / sqrt(d)  # [1, seq_len+1]
    weights = softmax(scores)
    output = weights @ V_cache  # [1, d]

    return output, K_cache, V_cache
```

## 2. KV Cache 内存计算

```python
def kv_cache_memory_formula(
    batch_size: int,     # B
    seq_length: int,     # S (当前序列长度)
    num_layers: int,     # L
    num_kv_heads: int,   # H_kv (KV 头数，GQA 中 < num_heads)
    head_dim: int,       # d_h
    dtype_size: int = 2  # FP16 = 2 bytes
) -> dict:
    """
    KV Cache 内存占用 = 2 × B × S × L × H_kv × d_h × dtype_size
    因子 2 是因为有 K 和 V 两部分
    """
    bytes_per_element = dtype_size
    total_elements = 2 * batch_size * seq_length * num_kv_heads * head_dim * num_layers
    total_bytes = total_elements * bytes_per_element

    return {
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024**3),
        "per_token_bytes": 2 * num_kv_heads * head_dim * num_layers * bytes_per_element,
        "formula": f"2 × B({batch_size}) × S({seq_length}) × L({num_layers}) × "
                   f"H_kv({num_kv_heads}) × d_h({head_dim}) × {bytes_per_element}"
    }

# 不同模型的 KV Cache 计算示例
models = {
    "LLaMA-7B": {"layers": 32, "kv_heads": 32, "head_dim": 128},
    "LLaMA-70B": {"layers": 80, "kv_heads": 8, "head_dim": 128},  # GQA
    "Mistral-7B": {"layers": 32, "kv_heads": 8, "head_dim": 128},  # GQA
}

for name, cfg in models.items():
    result = kv_cache_memory_formula(1, 4096, cfg["layers"], cfg["kv_heads"], cfg["head_dim"])
    print(f"{name} (B=1, S=4096): {result['total_gb']:.2f} GB")
```

```
计算结果:
  LLaMA-7B:  B=1, S=4096 → ~4 GB
  LLaMA-70B: B=1, S=4096 → ~4 GB (GQA 大幅减少!)
  Mistral-7B: B=1, S=4096 → ~1 GB (GQA)

GQA 的优势: 将 KV heads 从 32 减少到 8，KV Cache 缩小 4 倍
```

## 3. KV Cache 动态分配

```python
import torch

class KVCacheManager:
    """KV Cache 动态管理器"""

    def __init__(self, num_layers: int, num_kv_heads: int,
                 head_dim: int, max_seq_len: int = 4096,
                 dtype: torch.dtype = torch.float16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # 预分配最大空间
        self.cache_shape = (
            num_layers, 2, num_kv_heads, max_seq_len, head_dim
        )
        self.kv_cache = torch.zeros(self.cache_shape, dtype=dtype)
        self.current_seq_len = 0

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """追加新的 KV 对"""
        # new_k, new_v: [num_layers, num_kv_heads, 1, head_dim]
        if self.current_seq_len >= self.max_seq_len:
            raise RuntimeError("KV Cache 已满")

        self.kv_cache[:, 0, :, self.current_seq_len, :] = new_k.squeeze(2)
        self.kv_cache[:, 1, :, self.current_seq_len, :] = new_v.squeeze(2)
        self.current_seq_len += 1

    def get_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 KV 缓存"""
        seq_len = self.current_seq_len
        k = self.kv_cache[layer_idx, 0, :, :seq_len, :]  # [H_kv, S, d_h]
        v = self.kv_cache[layer_idx, 1, :, :seq_len, :]
        return k, v

    def get_memory_usage(self) -> dict:
        return {
            "allocated_gb": self.kv_cache.element_size() * self.kv_cache.nelement() / 1024**3,
            "used_gb": self.kv_cache.element_size() * 2 * self.current_seq_len *
                       self.num_kv_heads * self.head_dim * self.num_layers / 1024**3,
            "utilization": self.current_seq_len / self.max_seq_len
        }
```

## 4. Prefix Caching

```python
class PrefixKVCache:
    """前缀缓存：相同前缀的请求共享 KV Cache"""

    def __init__(self):
        self.prefix_cache = {}  # hash -> KV Cache tensor

    def get_or_compute(self, prefix_tokens: list, compute_fn) -> torch.Tensor:
        """获取或计算前缀的 KV Cache"""
        prefix_hash = hash(tuple(prefix_tokens))

        if prefix_hash in self.prefix_cache:
            return self.prefix_cache[prefix_hash].clone()

        # 计算并缓存
        kv = compute_fn(prefix_tokens)
        self.prefix_cache[prefix_hash] = kv
        return kv.clone()

# 使用场景: 多个请求共享相同的 system prompt
# System Prompt: 1000 tokens
# 请求 A: [system_prompt] + "用户问题A"
# 请求 B: [system_prompt] + "用户问题B"
# system_prompt 部分的 KV Cache 只需计算一次
```

## 5. KV Cache 量化

```python
class QuantizedKVCache:
    """量化 KV Cache 以减少内存占用"""

    def __init__(self, quantization: str = "int8"):
        self.quant_type = quantization

    def quantize(self, tensor: torch.Tensor) -> tuple:
        """将 FP16 KV 缓存量化"""
        if self.quant_type == "int8":
            scale = tensor.abs().max() / 127
            quantized = (tensor / scale).round().to(torch.int8)
            return quantized, scale
        elif self.quant_type == "fp8":
            return tensor.to(torch.float8_e4m3fn), None

    def dequantize(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        """反量化"""
        if self.quant_type == "int8":
            return quantized.to(torch.float16) * scale
        return quantized.to(torch.float16)

# 效果: INT8 KV Cache 比 FP16 减少 50% 内存
# FP8 KV Cache 减少 50% 内存，精度损失更小
```

## 6. Token 淘汰策略

```python
class KVCacheEviction:
    """长序列 KV Cache 淘汰策略"""

    def __init__(self, max_cache_len: int = 4096):
        self.max_len = max_cache_len

    def sliding_window(self, kv_cache: tuple, window_size: int = 4096) -> tuple:
        """滑动窗口：只保留最近的 token"""
        k, v = kv_cache
        if k.shape[1] > window_size:
            k = k[:, -window_size:, :]
            v = v[:, -window_size:, :]
        return k, v

    def streaming_llm(self, kv_cache: tuple, sink_size: int = 4,
                       window_size: int = 2048) -> tuple:
        """
        StreamingLLM 策略:
        保留开头的 sink tokens + 最近的 window tokens
        """
        k, v = kv_cache
        if k.shape[1] <= sink_size + window_size:
            return k, v

        # 保留 sink + 最近 window
        k_sinks = k[:, :sink_size, :]
        k_recent = k[:, -(window_size - sink_size):, :]
        v_sinks = v[:, :sink_size, :]
        v_recent = v[:, -(window_size - sink_size):, :]

        k_new = torch.cat([k_sinks, k_recent], dim=1)
        v_new = torch.cat([v_sinks, v_recent], dim=1)

        return k_new, v_new

    def attention_based_eviction(self, kv_cache: tuple,
                                  attention_scores: torch.Tensor,
                                  keep_ratio: float = 0.5) -> tuple:
        """基于注意力分数淘汰不重要的 token"""
        k, v = kv_cache
        seq_len = k.shape[1]

        # 计算每个 token 的平均被关注度
        avg_attention = attention_scores.mean(dim=0)  # [seq_len]

        # 保留关注度最高的 token
        keep_count = max(int(seq_len * keep_ratio), 64)
        _, keep_indices = torch.topk(avg_attention, keep_count)
        keep_indices = keep_indices.sort()[0]

        return k[:, keep_indices, :], v[:, keep_indices, :]
```

## 总结

KV Cache 是大模型推理**最关键的优化技术**，将 Decode 阶段的计算从 O(n^2) 降为 O(n)。但 KV Cache 本身消耗大量 GPU 内存，管理策略（动态分配、前缀缓存、量化、淘汰）直接影响推理系统的可扩展性和效率。
