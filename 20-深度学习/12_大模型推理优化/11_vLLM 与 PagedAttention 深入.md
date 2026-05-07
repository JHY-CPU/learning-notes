# 11_vLLM 与 PagedAttention 深入

## 1. vLLM 简介

vLLM 是 UC Berkeley 开发的**高吞吐 LLM 推理引擎**，核心创新是 PagedAttention -- 借鉴操作系统虚拟内存的分页思想管理 KV Cache。

```
vLLM 核心技术栈:

┌─────────────────────────────────┐
│          API 层                  │
│  OpenAI 兼容 API / Python SDK   │
├─────────────────────────────────┤
│          调度层                  │
│  Continuous Batching Scheduler   │
├─────────────────────────────────┤
│         注意力层                 │
│       PagedAttention            │
├─────────────────────────────────┤
│         内存管理层               │
│    Block Manager (虚拟内存)     │
├─────────────────────────────────┤
│          执行层                  │
│   Tensor Parallel / CUDA Graph  │
└─────────────────────────────────┘
```

## 2. PagedAttention 原理

### 2.1 问题：传统 KV Cache 的内存浪费

```
传统 KV Cache 的问题:

1. 预分配最大长度
   请求 A: 分配 2048 tokens 空间，实际用 100 tokens → 浪费 95%
   请求 B: 分配 2048 tokens 空间，实际用 2000 tokens

2. 内存碎片
   [请求A][  空闲  ][请求B][空闲][请求C][  空闲  ]
   无法利用空闲区域分配给新请求

3. 内存不可共享
   多个请求的相同前缀 (如 system prompt) 无法共享 KV Cache
```

### 2.2 PagedAttention 的解决

```
PagedAttention: 类似操作系统虚拟内存

逻辑视图 (每个请求看到连续内存):
  Request A: [Block0][Block1][Block2]
  Request B: [Block0][Block1]

物理视图 (GPU 实际内存, 按 block 分配):
  [Block池]
  Block 0: A 的前 16 tokens
  Block 1: B 的前 16 tokens
  Block 2: A 的 17-32 tokens
  Block 3: 空闲
  Block 4: B 的 17-32 tokens
  Block 5: A 的 33-48 tokens
  ...

Block Table (映射表):
  Request A → [0, 2, 5]
  Request B → [1, 4]

优势:
  ✓ 无预分配浪费 (按需分配)
  ✓ 无碎片 (block 粒度管理)
  ✓ 可共享 (copy-on-write)
```

## 3. Block Manager 实现

```python
class Block:
    """KV Cache 数据块"""

    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0        # 引用计数
        self.filled_slots = 0     # 已使用 slot 数

        # 实际 KV 数据
        self.key_cache = None     # [num_layers, num_kv_heads, block_size, head_dim]
        self.value_cache = None

    @property
    def is_full(self):
        return self.filled_slots >= self.block_size

    @property
    def free_slots(self):
        return self.block_size - self.filled_slots


class BlockManager:
    """Block 分配与管理"""

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int,
                 block_size: int = 16):
        self.block_size = block_size

        # GPU block 池
        self.gpu_blocks = {
            i: Block(i, block_size) for i in range(num_gpu_blocks)
        }
        self.free_gpu_blocks = list(range(num_gpu_blocks))

        # CPU block 池 (swap 用)
        self.cpu_blocks = {
            i + num_gpu_blocks: Block(i + num_gpu_blocks, block_size)
            for i in range(num_cpu_blocks)
        }
        self.free_cpu_blocks = list(range(num_gpu_blocks, num_gpu_blocks + num_cpu_blocks))

        # 请求的 block 分配记录
        self.request_blocks: dict[str, list[int]] = {}

    def allocate(self, request_id: str, num_tokens: int) -> list[int]:
        """为请求分配 blocks"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_gpu_blocks) < num_blocks:
            # 尝试驱逐其他请求
            self.try_evict(num_blocks)

        if len(self.free_gpu_blocks) < num_blocks:
            raise RuntimeError("GPU 内存不足")

        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_gpu_blocks.pop(0)
            self.gpu_blocks[block_id].ref_count = 1
            allocated.append(block_id)

        self.request_blocks[request_id] = allocated
        return allocated

    def append_token(self, request_id: str):
        """为请求追加 token"""
        blocks = self.request_blocks[request_id]
        last_block = self.gpu_blocks[blocks[-1]]

        if last_block.is_full:
            # 分配新 block
            if not self.free_gpu_blocks:
                self.try_evict(1)
            new_id = self.free_gpu_blocks.pop(0)
            self.gpu_blocks[new_id].ref_count = 1
            blocks.append(new_id)
        else:
            last_block.filled_slots += 1

    def free(self, request_id: str):
        """释放请求的所有 blocks"""
        if request_id not in self.request_blocks:
            return

        for block_id in self.request_blocks[request_id]:
            block = self.gpu_blocks.get(block_id) or self.cpu_blocks.get(block_id)
            block.ref_count -= 1
            if block.ref_count == 0:
                if block_id in self.free_gpu_blocks:
                    pass
                else:
                    self.free_gpu_blocks.append(block_id)

        del self.request_blocks[request_id]

    def swap(self, request_id: str, to_device: str):
        """GPU ↔ CPU swap"""
        blocks = self.request_blocks[request_id]
        src_pool = self.gpu_blocks if to_device == "cpu" else self.cpu_blocks
        dst_pool = self.cpu_blocks if to_device == "cpu" else self.gpu_blocks
        src_free = self.free_gpu_blocks if to_device == "cpu" else self.free_cpu_blocks
        dst_free = self.free_cpu_blocks if to_device == "cpu" else self.free_gpu_blocks

        new_blocks = []
        for old_id in blocks:
            # 释放源 block
            src_pool[old_id].ref_count -= 1
            src_free.append(old_id)

            # 分配目标 block
            new_id = dst_free.pop(0)
            dst_pool[new_id].ref_count = 1
            new_blocks.append(new_id)

        self.request_blocks[request_id] = new_blocks
```

## 4. PagedAttention 计算

```python
class PagedAttentionKernel:
    """PagedAttention CUDA Kernel (伪代码)"""

    @staticmethod
    def forward(query, key_cache, value_cache, block_table, context_lens):
        """
        query: [batch, num_heads, 1, head_dim]
        key_cache: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
        value_cache: 同上
        block_table: [batch, max_blocks_per_seq]
        context_lens: [batch]
        """
        batch_size = query.shape[0]
        outputs = []

        for b in range(batch_size):
            q = query[b]  # [num_heads, 1, head_dim]
            blocks = block_table[b]  # 该请求的 block 列表
            seq_len = context_lens[b]

            # 按 block 读取 KV Cache
            k_parts = []
            v_parts = []
            remaining = seq_len

            for block_id in blocks:
                if remaining <= 0:
                    break
                k = key_cache[block_id]  # [num_kv_heads, block_size, head_dim]
                v = value_cache[block_id]

                # 只取实际使用的部分
                take = min(remaining, k.shape[1])
                k_parts.append(k[:, :take, :])
                v_parts.append(v[:, :take, :])
                remaining -= take

            K = torch.cat(k_parts, dim=1)  # [num_kv_heads, seq_len, head_dim]
            V = torch.cat(v_parts, dim=1)

            # 计算注意力
            scores = torch.matmul(q, K.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
            outputs.append(out)

        return torch.stack(outputs)
```

## 5. vLLM 使用

```python
from vllm import LLM, SamplingParams

# 基础使用
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    dtype="float16",
)

# 批量生成
prompts = ["解释量子计算", "什么是深度学习"]
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

outputs = llm.generate(prompts, params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")

# OpenAI 兼容 API 服务
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-2-7b-hf \
#     --host 0.0.0.0 --port 8000
```

## 6. vLLM 性能调优

```python
# 性能调优参数
llm = LLM(
    model="model",

    # 内存管理
    gpu_memory_utilization=0.92,    # GPU 内存利用率
    swap_space=4,                    # CPU swap 空间 (GB)

    # 批处理
    max_num_batched_tokens=8192,    # 最大批处理 token 数
    max_num_seqs=256,               # 最大并发序列

    # KV Cache
    block_size=16,                   # Block 大小
    enable_prefix_caching=True,     # 启用前缀缓存

    # 并行
    tensor_parallel_size=1,         # 张量并行度

    # 优化
    use_v2_block_manager=True,      # V2 Block 管理器
    enable_chunked_prefill=True,    # 分块 Prefill
)

# 性能参考 (A100 80GB):
# LLaMA-2 7B:
#   吞吐量: ~3000 tokens/s (batch=64)
#   首 token: ~50ms
#   单 token: ~8ms
```

## 总结

vLLM 通过 PagedAttention 将 KV Cache 管理从**预分配**变为**按需分页**，消除了内存碎片和浪费，配合连续批处理实现了**业界领先的推理吞吐量**。对于大规模 LLM 部署，vLLM 是当前首选的推理引擎。
