# 9_Pipeline Parallelism 流水线并行

## 1. 流水线并行概述

流水线并行 (Pipeline Parallelism, PP) 将**模型的不同层分配到不同 GPU**，数据像流水线一样依次通过各阶段。

```
PP vs TP:

张量并行 (TP):           流水线并行 (PP):
  同一层跨多个 GPU          不同层在不同 GPU
  高频通信                  低频通信
  需要 NVLink               可用 PCIe
  GPU 间紧密耦合            GPU 间松耦合

PP 划分示例 (32 层模型, 4 GPU):
  GPU 0: Layer 0-7
  GPU 1: Layer 8-15
  GPU 2: Layer 16-23
  GPU 3: Layer 24-31
```

## 2. Naive 流水线 (问题方案)

```
Naive 流水线 (有严重的气泡问题):

时间 →
GPU 0: [MB0 ████]          [MB1 ████]          [MB2 ████]
GPU 1:         [MB0 ████]          [MB1 ████]          [MB2 ████]
GPU 2:                 [MB0 ████]          [MB1 ████]          [MB2 ████]
GPU 3:                         [MB0 ████]          [MB1 ████]          [MB2 ████]

█ = 计算, 空白 = 空闲 (气泡)
气泡率 = (N-1) / (N + M-1) ≈ (N-1)/M
  N = 流水线阶段数
  M = 微批次数量

问题: GPU 大部分时间在等待! 利用率极低。
```

## 3. 1F1B 调度 (PipeDream)

```
1F1B (One Forward One Backward):

时间 →
GPU 0: F0 F1 F2 F3 F4 F5 F6 F7  B0 B1 B2 B3 B4 B5 B6 B7
GPU 1:    F0 F1 F2 F3 F4 F5 F6 F7  B0 B1 B2 B3 B4 B5 B6 B7
GPU 2:       F0 F1 F2 F3 F4 F5 F6 F7  B0 B1 B2 B3 B4 B5 B6 B7
GPU 3:          F0 F1 F2 F3 F4 F5 F6 F7  B0 B1 B2 B3 B4 B5 B6 B7

F = Forward, B = Backward
数字 = 微批次编号

优点: 交替执行前向和后向，减少气泡
内存: 每个 GPU 只需存储其负责层的激活值
```

## 4. Interleaved 1F1B (Megatron-LM)

```python
class InterleavedPipeline:
    """交错流水线调度"""

    def __init__(self, num_stages, num_micro_batches):
        self.num_stages = num_stages
        self.num_mb = num_micro_batches

    def schedule(self):
        """
        每个 GPU 负责多个非连续的层组
        GPU 0: Layer 0-1, 8-9, 16-17, 24-25
        GPU 1: Layer 2-3, 10-11, 18-19, 26-27
        ...

        减少气泡: 气泡率从 (N-1)/M 降至 (N-1)/(V×M)
        V = 虚拟阶段数 (每 GPU 负责的层组数)
        """
        schedule = []
        V = 4  # 虚拟阶段数

        for mb in range(self.num_mb):
            for v in range(V):
                for stage in range(self.num_stages):
                    # Forward
                    schedule.append({
                        "type": "F",
                        "micro_batch": mb,
                        "virtual_stage": v,
                        "stage": stage
                    })

        for mb in range(self.num_mb):
            for v in range(V - 1, -1, -1):
                for stage in range(self.num_stages - 1, -1, -1):
                    # Backward
                    schedule.append({
                        "type": "B",
                        "micro_batch": mb,
                        "virtual_stage": v,
                        "stage": stage
                    })

        return schedule
```

## 5. 气泡率分析

```python
def pipeline_bubble_analysis(num_stages, num_micro_batches, virtual_stages=1):
    """
    气泡率分析

    Naive PP:    气泡率 = (N - 1) / (N + M - 1)
    1F1B:        气泡率 = (N - 1) / M
    Interleaved: 气泡率 = (N - 1) / (V * M)

    N = 流水线阶段数
    M = 微批次数
    V = 虚拟阶段数
    """
    N = num_stages
    M = num_micro_batches
    V = virtual_stages

    naive_bubble = (N - 1) / (N + M - 1)
    onefb_bubble = (N - 1) / M
    interleaved_bubble = (N - 1) / (V * M)

    print(f"流水线阶段数: {N}")
    print(f"微批次数: {M}")
    print(f"虚拟阶段数: {V}")
    print(f"Naive 气泡率: {naive_bubble:.1%}")
    print(f"1F1B 气泡率: {onefb_bubble:.1%}")
    print(f"Interleaved 气泡率: {interleaved_bubble:.1%}")
    print(f"GPU 利用率 (Interleaved): {1 - interleaved_bubble:.1%}")

# 示例
pipeline_bubble_analysis(num_stages=8, num_micro_batches=64, virtual_stages=4)
"""
流水线阶段数: 8
微批次数: 64
虚拟阶段数: 4
Naive 气泡率: 10.3%
1F1B 气泡率: 10.9%
Interleaved 气泡率: 2.7%
GPU 利用率 (Interleaved): 97.3%
"""
```

## 6. PP 在推理中的应用

```python
class PipelineInference:
    """推理时的流水线并行"""

    def __init__(self, model_layers, num_stages):
        self.stages = self.partition_layers(model_layers, num_stages)
        self.num_stages = num_stages

    def partition_layers(self, layers, num_stages):
        """将模型层分配到各阶段"""
        layers_per_stage = len(layers) // num_stages
        stages = []
        for i in range(num_stages):
            start = i * layers_per_stage
            end = start + layers_per_stage
            stages.append(layers[start:end])
        return stages

    def generate(self, input_ids, max_tokens):
        """
        推理时的流水线执行

        Prefill: 可以流水线化 (batch 处理)
        Decode:  难以流水线化 (逐 token)
        """
        # Prefill 阶段 - 流水线化
        hidden = self.pipeline_forward(input_ids)

        # Decode 阶段 - 通常用 TP
        tokens = []
        for _ in range(max_tokens):
            # 单 token 生成时 PP 效率低
            # 通常用 TP + KV Cache
            hidden = self.pipeline_forward_single(hidden[:, -1:])
            next_token = self.decode_token(hidden)
            tokens.append(next_token)

        return tokens
```

## 7. TP + PP 组合

```
Megatron-LM 推荐的并行策略:

TP × PP = 总 GPU 数

示例: 64 GPU 训练 LLaMA-70B
  方案 A: TP=8, PP=8  → 8卡 TP, 8个 PP 阶段
  方案 B: TP=4, PP=16 → 4卡 TP, 16个 PP 阶段
  方案 C: TP=2, PP=32 → 2卡 TP, 32个 PP 阶段

选择原则:
  1. TP 度尽可能小 (通信开销)
  2. TP 度必须能整除模型维度
  3. PP 阶段数用剩余 GPU 数
  4. 推荐 TP ≤ 8 (NVLink 限制)
```

## 8. 内存分布

```
PP 内存分布:

每个 GPU 存储:
  1. 模型权重: W/N (N = PP 阶段数)
  2. 优化器状态: O/N (训练时)
  3. 激活值: 与其负责层相关

推理时:
  GPU 0: Layer 0-7 权重 + 前几层 KV Cache
  GPU 1: Layer 8-15 权重 + 中间层 KV Cache
  ...

优势: 单 GPU 内存需求 = 总模型大小 / PP 阶段数
```

## 总结

流水线并行将模型层分布到不同 GPU，通过微批次调度减少气泡。1F1B 和 Interleaved 调度可以将 GPU 利用率提升到 90%+。PP 与 TP 组合使用是训练超大模型的标准方案。在推理场景，PP 主要用于 Prefill 阶段，Decode 阶段更依赖 TP。
