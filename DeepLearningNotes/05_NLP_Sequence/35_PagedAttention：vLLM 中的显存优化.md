# 35_PagedAttention：vLLM 中的显存优化

## 核心概念
- **PagedAttention**：由 Kwon et al. (2023) 提出，是 vLLM 推理引擎的核心技术。灵感来自操作系统中的虚拟内存分页（paging），解决 LLM 推理时的 KV cache 显存管理问题。
- **KV cache (键值缓存)**：在自回归生成中，每个 token 的 Key 和 Value 矩阵会在所有后续步骤中被复用。将它们缓存起来避免重复计算，但显存占用随序列长度线性增长。
- **显存碎片化问题**：传统的 KV cache 为整个序列分配连续的显存块，但变长序列导致显存碎片化严重，实际利用率仅 20-60%。
- **分页管理**：将 KV cache 分成固定大小的块（pages/block），每个块存储固定数量（如 16 个）token 的 KV 值。序列的 KV cache 由非连续的物理块组成，通过逻辑到物理的映射表管理。
- **逻辑页 vs 物理页**：逻辑视图是连续的 token 序列，物理视图是分散的显存块。这种解耦使得显存分配更加灵活，消除了碎片化。
- **Copy-on-Write (写时复制)**：当多个序列共享前缀（如 beam search 或并行采样），PagedAttention 通过指针共享物理块，在发生写操作时才复制。节省大量显存。
- **vLLM 推理引擎**：基于 PagedAttention 构建的高吞吐推理系统，相比 FasterTransformer、TensorRT-LLM 等传统方案，吞吐量提升 2-4 倍。

## 数学推导
KV cache 的显存占用：
$$
\text{Memory}_{KV} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch\_size} \times \text{dtype\_bytes}
$$

**分页管理**：设块大小为 $B$ tokens，序列长度为 $L$，需要的块数：
$$
N_{\text{blocks}} = \lceil L / B \rceil
$$

物理块总数：$N_{\text{total\_blocks}} = \lfloor \text{total\_GPU\_memory} / \text{block\_size} \rfloor$

**显存利用率**：
$$
\text{Utilization} = \frac{\sum \text{有效 KV cache 大小}}{\text{分配的总显存}}
$$

PagedAttention 通过消除碎片化，使得利用率接近 100%（受限于块内碎片，约 $(B-1)/B$ 的最小浪费）。

## 直观理解
- **PagedAttention 像操作系统的虚拟内存**：你打开一个很大的文件，操作系统并未将整个文件加载到物理内存，而是只在需要时加载"页"。PagedAttention 同理——KV cache 分散存储在非连续的"页"中，通过"页表"管理，极大减少了碎片。
- **普通 KV cache 像连续排列的图书馆书架**：每个序列必须占满一整排书架（连续显存）。如果序列 A 用 3 排，序列 B 用 5 排，但中间有空隙——空隙无法被序列 C 利用（碎片化）。
- **PagedAttention 像分散的共享储物柜**：每个序列只拿需要的储物格（pages），这些格子在空间上可以任意分散。更大的序列拿更多格，更小的序列释放格子。完全消除了"书架空隙"浪费。
- **Copy-on-Write 像"共享课本"**：两个学生（两个采样序列）看同一本教材的前三章（共享物理块），当其中一个人要做笔记时，才复印自己需要的页面（写时复制）。

## 代码示例
```python
import torch

# 模拟 PagedAttention 的分页管理
class PagedKVManager:
    """简化的 KV cache 分页管理器"""
    def __init__(self, total_pages, page_size=16, dim=128):
        self.total_pages = total_pages
        self.page_size = page_size      # 每页存储的 token 数
        self.dim = dim                  # 每个 token 的维度
        # 物理页表
        self.k_cache = torch.zeros(total_pages, page_size, dim)
        self.v_cache = torch.zeros(total_pages, page_size, dim)
        self.free_pages = set(range(total_pages))
        # 逻辑到物理的映射
        self.seq_tables = {}            # seq_id -> [page_id, ...]

    def alloc_pages(self, seq_id, num_pages):
        if len(self.free_pages) < num_pages:
            raise OOMError("显存不足！")
        allocated = []
        for _ in range(num_pages):
            page = self.free_pages.pop()
            allocated.append(page)
        self.seq_tables[seq_id] = allocated

    def write(self, seq_id, token_pos, k, v):
        seq_pages = self.seq_tables[seq_id]
        page_idx = token_pos // self.page_size
        offset = token_pos % self.page_size
        physical_page = seq_pages[page_idx]
        self.k_cache[physical_page, offset] = k
        self.v_cache[physical_page, offset] = v

    def read(self, seq_id, token_pos):
        seq_pages = self.seq_tables[seq_id]
        page_idx = token_pos // self.page_size
        offset = token_pos % self.page_size
        physical_page = seq_pages[page_idx]
        return self.k_cache[physical_page, offset], self.v_cache[physical_page, offset]

class OOMError(Exception):
    pass

# 模拟 PagedAttention 管理
manager = PagedKVManager(total_pages=1024, page_size=16, dim=128)
batch = 8
seq_len = 256

# 每个序列需要的页数
pages_per_seq = (seq_len + 15) // 16  # 256/16 = 16
print(f"每个序列需要 {pages_per_seq} 页，共 {batch * pages_per_seq} 页")

# 分配页面
for i in range(batch):
    manager.alloc_pages(i, pages_per_seq)

# 模拟写入
for i in range(batch):
    for pos in range(seq_len):
        k = torch.randn(128)
        v = torch.randn(128)
        manager.write(i, pos, k, v)
print("KV cache 写入完成")
print(f"剩余空闲页: {len(manager.free_pages)}")
```

## 深度学习关联
- **LLM 推理的显存瓶颈**：KV cache 是 LLM 推理的主要显存消耗者（对于长序列，KV cache 可占总显存的 60-90%）。PagedAttention 通过消除碎片化和管理优化显著提升推理吞吐。
- **连续批处理 (Continuous Batching)**：vLLM 将 PagedAttention 与迭代级调度结合，支持在一个 batch 中动态添加/移除序列，进一步提升硬件利用率。
- **KV cache 复用技术**：PagedAttention 的共享前缀机制催生了 Prefix Caching（共享 prompt 前缀的 KV cache）等多种优化，在 RAG 和多轮对话中特别有效。
