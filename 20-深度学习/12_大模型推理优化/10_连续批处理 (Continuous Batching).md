# 10_连续批处理 (Continuous Batching)

## 1. 批处理策略对比

### 1.1 静态批处理 (Static Batching)

```
静态批处理: 等待所有请求完成后再处理下一批

时间 →
请求 A: [████████████████]  (生成 16 tokens)
请求 B: [████████]          (生成 8 tokens)
请求 C: [████████████████████████]  (生成 24 tokens)
        └────── 等待 ──────┘
                     ↑ 请求 B 已完成但必须等待

问题:
  - 短请求被长请求阻塞 (Head-of-line blocking)
  - GPU 利用率低 (等待期间 GPU 空闲)
  - 吞吐量受限于最长请求
```

### 1.2 连续批处理 (Continuous Batching / Iteration-level Batching)

```
连续批处理: 每个 iteration 都可以加入新请求

时间 →
Iter 1: [A, B, C]  → 生成 token
Iter 2: [A, B, C]  → 生成 token
Iter 3: [A, B, C]  → B 完成, 加入 D → [A, C, D]
Iter 4: [A, C, D]  → 生成 token
Iter 5: [A, C, D]  → A 完成, 加入 E → [C, D, E]
...

优势:
  - 无 Head-of-line blocking
  - GPU 始终满载
  - 吞吐量提升 2-10x
```

## 2. 连续批处理核心实现

```python
class ContinuousBatchingScheduler:
    """连续批处理调度器"""

    def __init__(self, model, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.running_requests = []  # 当前批处理中的请求
        self.waiting_queue = []      # 等待队列

    def add_request(self, request):
        """添加新请求"""
        self.waiting_queue.append(request)

    def step(self):
        """执行一步推理 (一个 token 的生成)"""
        # 1. 检查完成的请求，移出批处理
        self.running_requests = [
            req for req in self.running_requests
            if not req.is_finished()
        ]

        # 2. 从等待队列补充新请求
        while (len(self.running_requests) < self.max_batch_size
               and self.waiting_queue):
            new_req = self.waiting_queue.pop(0)
            new_req.start_time = time.time()
            self.running_requests.append(new_req)

        if not self.running_requests:
            return

        # 3. 批量前向传播 (每个请求生成 1 个 token)
        batch_tokens = self.prepare_batch()
        logits = self.model.forward_batch(batch_tokens)

        # 4. 采样并更新各请求
        for i, req in enumerate(self.running_requests):
            next_token = self.sample(logits[i])
            req.generated_tokens.append(next_token)
            req.step_count += 1

    def prepare_batch(self):
        """准备批处理输入（处理不同长度）"""
        max_len = max(req.current_length for req in self.running_requests)

        batch = []
        for req in self.running_requests:
            # Padding 到相同长度
            padded = req.tokens + [0] * (max_len - req.current_length)
            batch.append(padded)

        return torch.tensor(batch)
```

## 3. Paged KV Cache 与连续批处理

```python
class PagedKVCacheManager:
    """
    PagedAttention 的 KV Cache 管理
    支持连续批处理的动态内存分配
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 2048):
        self.block_size = block_size  # 每个 block 存储的 token 数
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.block_table = {}  # request_id -> [block_ids]

    def allocate(self, request_id: str, num_tokens: int):
        """为请求分配 KV Cache 空间"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("GPU 内存不足")

        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        self.block_table[request_id] = allocated

    def append_token(self, request_id: str):
        """为请求追加一个 token 的 KV Cache"""
        blocks = self.block_table[request_id]
        last_block = blocks[-1]

        # 检查最后一个 block 是否已满
        if self.get_block_usage(last_block) >= self.block_size:
            # 需要分配新 block
            if not self.free_blocks:
                raise RuntimeError("GPU 内存不足")
            new_block = self.free_blocks.pop(0)
            blocks.append(new_block)

    def free(self, request_id: str):
        """释放请求的 KV Cache"""
        if request_id in self.block_table:
            for block in self.block_table[request_id]:
                self.free_blocks.append(block)
            del self.block_table[request_id]

    def get_kv_for_request(self, request_id: str, layer_idx: int):
        """获取指定请求的 KV Cache"""
        blocks = self.block_table[request_id]
        # 从分散的 blocks 中组装 KV Cache
        k_parts = []
        v_parts = []
        for block_id in blocks:
            k, v = self.get_block_kv(block_id, layer_idx)
            k_parts.append(k)
            v_parts.append(v)

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)
```

## 4. 调度策略

```python
class SchedulingStrategies:
    """不同的请求调度策略"""

    # FCFS (First Come First Served)
    def fcfs(self, waiting_queue):
        """先来先服务"""
        return waiting_queue.pop(0)

    # SJF (Shortest Job First)
    def sjf(self, waiting_queue):
        """短作业优先"""
        waiting_queue.sort(key=lambda r: r.max_new_tokens)
        return waiting_queue.pop(0)

    # Priority-based
    def priority(self, waiting_queue):
        """优先级调度"""
        waiting_queue.sort(key=lambda r: -r.priority)
        return waiting_queue.pop(0)

    # Preemptive (抢占式)
    def preemptive(self, running, waiting, max_running):
        """
        抢占式调度:
        高优先级请求可以抢占正在运行的低优先级请求
        被抢占的请求保存状态，稍后恢复
        """
        if len(running) >= max_running:
            lowest = min(running, key=lambda r: r.priority)
            if waiting[0].priority > lowest.priority:
                lowest.save_state()
                running.remove(lowest)
                return waiting.pop(0)
        return None
```

## 5. 性能分析

```python
"""
连续批处理性能提升分析:

场景: 100 个请求，平均生成 50 tokens

静态批处理:
  - 批大小: 8
  - 每批处理时间: max(请求长度) × 单 token 延迟
  - 假设最长请求 200 tokens, 单 token 10ms
  - 每批: 200 × 10ms = 2s
  - 总批次数: 100/8 = 13
  - 总时间: 13 × 2s = 26s
  - 吞吐量: 100 × 50 / 26 = 192 tokens/s

连续批处理:
  - 每 iteration 处理所有运行中的请求 (1 个 token)
  - 假设平均同时运行 8 个请求
  - 单 iteration: 10ms
  - 总 iteration: 100 × 50 = 5000
  - 总时间: 5000 × 10ms = 50s? → 不对
  - 实际: GPU 始终 8 请求并行, 总 tokens = 5000
  - 总时间: 5000 / 8 × 10ms = 6.25s
  - 吞吐量: 5000 / 6.25 = 800 tokens/s

加速比: 800 / 192 ≈ 4.2x
"""
```

## 6. vLLM 的连续批处理

```python
from vllm import LLM, SamplingParams

# vLLM 内置连续批处理
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_batched_tokens=8192,   # 最大批处理 token 数
    max_num_seqs=64,               # 最大并发序列数
    scheduler_policy="fcfs",       # 调度策略
)

# 批量请求 - 自动连续批处理
prompts = ["问题1", "问题2", ..., "问题100"]
params = SamplingParams(temperature=0.7, max_tokens=256)

# vLLM 自动调度所有请求
outputs = llm.generate(prompts, params)

# 每个请求独立完成，无需等待
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated: {output.outputs[0].text[:100]}...")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")
```

## 总结

连续批处理是 LLM 服务化的核心优化，通过**消除 Head-of-line blocking**实现 2-10x 的吞吐量提升。结合 PagedAttention 的动态 KV Cache 管理，可以高效处理大量并发请求。vLLM 等推理框架已内置了这些优化。
