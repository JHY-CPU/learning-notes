# vLLM架构 - 模型推理与Serving

*PagedAttention、Continuous Batching、KV Cache管理等核心创新，vLLM如何成为LLM推理的事实标准*

性能对比（论文数据）

- 相比 HuggingFace Transformers: 吞吐量提升
   **14x-24x**
- 相比 HuggingFace TGI: 吞吐量提升
   **2.2x-3.5x**
- KV Cache利用率从 ~50% 提升到
   **<4% 浪费**
- 支持在同等硬件上服务
   **4x-6x 更多并发请求**

vLLM 调度器 (Scheduler)

- **Running队列**
   : 当前正在处理的请求
- **Waiting队列**
   : 等待GPU资源的新请求
- **Swapped队列**
   : 因显存不足被换出到CPU的请求
- **调度逻辑**
   :

   1. Running队列中的请求生成一个token
   2. 如果请求完成（遇到EOS或达到max_tokens），从Running中移除
   3. 如果有空闲KV Cache块，从Waiting中选择新请求加入Running
   4. 如果显存不足，将低优先级请求Swap到CPU

分块预填充机制

- 将长prompt的prefill切分为多个chunk（如每chunk 512 tokens）
- 每个调度step中，prefill chunk和decode请求
   **混合调度**
- 保证decode请求的延迟不会被长prefill阻塞
- 代价：prefill阶段的吞吐量略有下降（约5-10%）
- 收益：P99延迟显著降低（特别是有长输入请求时）


<!-- Converted from: 01_vLLM架构.html -->
