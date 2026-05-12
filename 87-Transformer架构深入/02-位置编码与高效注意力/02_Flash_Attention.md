# Flash Attention


## Flash Attention


#### 核心洞察


Flash Attention 并没有改变注意力的数学计算（输出完全相同），而是重新组织了计算顺序以最大化利用 GPU 内存层级（SRAM vs HBM）。其核心发现是：标准注意力实现的瓶颈不在算力（FLOPs），而在内存带宽（HBM 访问）。通过 IO-aware 的分块计算策略，Flash Attention 将 HBM 访问量从 O(n²) 降低到 O(n²d²/M)，其中 M 是 SRAM 大小。


## 1. 背景：GPU 内存层级


### 1.1 内存层级结构


| 层级 | 容量 | 带宽 | 角色 |
| --- | --- | --- | --- |
| SRAM（片上缓存） | ~20 MB（A100: 40MB） | ~19 TB/s | 快速计算暂存 |
| HBM（显存） | 40-80 GB（A100） | ~2 TB/s | 数据存储 |
| DRAM（主存） | 数百 GB | ~50 GB/s | 离线存储 |


> **Warning:** #### HBM 访问是瓶颈
>
>
> 标准注意力实现中，计算 QK^T^ 产生 n×n 的中间矩阵 S，计算 softmax 需要读写 S，再计算 SV 又需要读写注意力矩阵 A。这些 n×n 矩阵被反复从 HBM 读写。当 n=4096, d=128 时，n×n 的矩阵是 16M 个元素（64MB），远超 SRAM 容量，必须反复访问 HBM。


### 1.2 标准注意力的 HBM 访问分析


```
# 标准注意力的 HBM 访问模式
# 步骤1: 从 HBM 读取 Q, K → 计算 S = QK^T → 写入 HBM
# 步骤2: 从 HBM 读取 S → 计算 softmax(S) → 写入 HBM
# 步骤3: 从 HBM 读取 S, V → 计算 O = softmax(S)V → 写入 HBM

# 总 HBM 访问量:
# 读: Q(n*d) + K(n*d) + S(n*n) + S(n*n) + V(n*d) = 2*n*d + 2*n²
# 写: S(n*n) + S(n*n) + O(n*d) = 2*n² + n*d
# 合计: O(n²) 次 HBM 访问（n² 项主导）

# 当 n >> d 时（如 n=8K, d=128）, n²=64M >> nd=1M
# HBM 访问量约 3 × n² × sizeof(float) = 768 MB 的读写
```


## 2. FlashAttention-1（IO-aware 分块计算）


### 2.1 核心思想：Tiling（分块）


Flash Attention 将 Q、K、V 分成小块（blocks），每个块的大小设计为可以完全放入 SRAM。在 SRAM 中完成注意力计算后，只将最终结果写回 HBM，避免中间矩阵 S 和 A 的 HBM 读写。


$$
分块大小: Br × d 和 Bc × d
        其中 Br = ⌊M / (4d)⌋, Bc = ⌊M / (4d)⌋
        M 为 SRAM 容量, d 为 head 维度
$$


### 2.2 Online Softmax 算法


分块计算的关键难点是 softmax：它需要全局归一化（需要知道所有位置的值）。Flash Attention 使用在线 softmax（online softmax）算法来解决这个问题：


```
# 在线 softmax: 在单次遍历中计算 softmax，无需两次遍历
# 维护两个运行变量: 当前最大值 m 和归一化因子 l

def online_softmax(scores):
    """
    scores: 一个向量，但我们逐块处理
    维护: m = 当前最大值, l = exp(x_i - m) 的累积和
    """
    m = float('-inf')
    l = 0.0
    output = torch.zeros_like(scores)

    for block in scores.split(B):
        # 更新最大值
        m_new = max(m, block.max())

        # 修正之前的累积值
        correction = math.exp(m - m_new)
        l = l * correction
        output = output * correction  # 修正之前的输出

        # 累积当前块
        exp_block = torch.exp(block - m_new)
        l = l + exp_block.sum()
        output = output + exp_block  # 暂存非归一化的值

        m = m_new

    # 最终归一化
    return output / l
```


### 2.3 Flash Attention 算法伪代码


```
def flash_attention(Q, K, V, B_r, B_c):
    """
    Q: (n, d), K: (n, d), V: (n, d) 均在 HBM 中
    返回: O: (n, d) 写入 HBM
    """
    n = Q.shape[0]

    # 将 Q 分成 T_r = ceil(n/B_r) 个行块
    # 将 K, V 分成 T_c = ceil(n/B_c) 个列块
    # 初始化输出和在线 softmax 的运行变量
    O = torch.zeros(n, d)       # 最终输出（在 HBM 中）
    l = torch.zeros(n)          # 行归一化因子
    m = torch.full((n,), -inf)  # 行最大值

    for i in range(T_r):  # 遍历 Q 的块
        # 从 HBM 加载 Q_i 到 SRAM
        Q_i = Q[i*B_r:(i+1)*B_r]  # (B_r, d)

        # 初始化当前块的输出和运行变量
        O_i = torch.zeros(B_r, d)
        l_i = torch.zeros(B_r)
        m_i = torch.full((B_r,), -inf)

        for j in range(T_c):  # 遍历 K, V 的块
            # 从 HBM 加载 K_j, V_j 到 SRAM
            K_j = K[j*B_c:(j+1)*B_c]  # (B_c, d)
            V_j = V[j*B_c:(j+1)*B_c]  # (B_c, d)

            # 在 SRAM 中计算注意力分数
            S_ij = Q_i @ K_j.T / sqrt(d)  # (B_r, B_c)，不写回 HBM

            # 在线 softmax 更新
            m_ij = S_ij.max(dim=-1).values  # (B_r,)
            m_new = torch.maximum(m_i, m_ij)

            # 修正因子
            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))  # (B_r, B_c)
            correction_old = torch.exp(m_i - m_new)
            correction_new = torch.exp(m_ij - m_new)

            # 更新归一化因子
            l_i = l_i * correction_old + P_ij.sum(dim=-1)

            # 更新输出
            O_i = O_i * correction_old.unsqueeze(-1) + P_ij @ V_j

            # 更新最大值
            m_i = m_new

        # 归一化并写回 HBM
        O_i = O_i / l_i.unsqueeze(-1)
        O[i*B_r:(i+1)*B_r] = O_i

    return O
```


### 2.4 HBM 访问量分析


| 操作 | 标准注意力 | Flash Attention | 改进 |
| --- | --- | --- | --- |
| HBM 读取 | 2nd + 2n² | 2nd + 2n²d²/M | O(n²) → O(n²d²/M) |
| HBM 写入 | 2n² + nd | nd | 消除中间矩阵 |
| 中间矩阵存储 | O(n²) S 和 A | 无（全在 SRAM） | 无额外内存 |


#### 速度提升来源


Flash Attention 的加速主要来自减少 HBM 访问，而非减少 FLOPs。实际上 Flash Attention 的 FLOPs 与标准注意力相同，但由于 HBM 访问大幅减少（约 5-20 倍），整体运行速度提升 2-4 倍。更少的 HBM 访问也意味着更低的能耗和更好的 GPU 利用率。


## 3. FlashAttention-2（并行化优化）


### 3.1 相比 V1 的改进


FlashAttention-2（Dao, 2023）在 V1 基础上做了进一步优化：


- **更好的并行化**
   ：V1 在 Q 的块维度上并行（外循环），V2 将并行化放在 K/V 的块维度上（内循环），减少 GPU 线程束（warp）间的同步开销
- **减少非矩阵乘法运算**
   ：将 softmax 的 rescaling 操作尽可能与矩阵乘法融合，减少非 matmul 操作（GPU 上 matmul 效率最高）
- **更好的 warp 调度**
   ：让每个 warp 处理完整的 Q 块，避免 V1 中 warp 间的 rescaling 同步


### 3.2 算法结构对比


```
# Flash Attention V1: 外循环遍历 Q 块
for i in range(T_r):        # Q 块
    for j in range(T_c):    # K/V 块
        compute_block(Q_i, K_j, V_j)

# Flash Attention V2: 内循环遍历 K/V 块，外循环遍历 Q 块
# 关键变化: 每个 warp 独立处理一个 Q 块的所有 K/V 块
# 避免了 warp 间的 rescaling 同步
for j in range(T_c):        # K/V 块
    # 每个 warp 并行处理不同的 Q 块
    for i in range(T_r):    # Q 块（在不同 warp 间并行）
        compute_block(Q_i, K_j, V_j)
    # softmax rescaling 在此统一处理
```


### 3.3 性能提升


| 序列长度 | FlashAttention-1 | FlashAttention-2 | V2 提升 |
| --- | --- | --- | --- |
| 512 | ~115 TFLOPS | ~150 TFLOPS | ~30% |
| 1024 | ~200 TFLOPS | ~250 TFLOPS | ~25% |
| 2048 | ~250 TFLOPS | ~300 TFLOPS | ~20% |
| 4096 | ~270 TFLOPS | ~330 TFLOPS | ~22% |


## 4. FlashAttention-3（硬件级优化）


### 4.1 针对 Hopper 架构的优化


FlashAttention-3（Dao et al., 2024）专门为 NVIDIA H100 GPU（Hopper 架构）做了深度优化：


- **异步流水线（Asynchronous Pipelining）**
   ：利用 H100 的 TMA（Tensor Memory Accelerator）异步数据搬运，在计算当前块的同时预取下一块数据
- **WGMMA 指令**
   ：利用 Hopper 新增的 Warpgroup MMA（wgmma.mma_async）指令，实现更高效的矩阵乘法
- **FP8 低精度支持**
   ：支持 FP8 (E4M3/E5M2) 数据类型，在保持精度的同时获得接近 2 倍的吞吐
- **非整数分块处理**
   ：对序列长度不是分块大小整数倍的情况做了优化，减少尾部浪费


### 4.2 低精度注意力


```
# FlashAttention-3 的 FP8 注意力
# 使用 blockwise 量化减少精度损失

def flash_attention_fp8(Q, K, V):
    """
    使用 FP8 进行注意力计算
    1. Q, K 以 FP8 存储和计算矩阵乘法
    2. 保留 softmax 计算在 FP32（对精度敏感）
    3. V 可以用 FP8，输出反量化回 BF16/FP16
    """
    # 每个 block 独立量化
    # Q_block_fp8 = quantize_fp8(Q_block, scale_q)
    # K_block_fp8 = quantize_fp8(K_block, scale_k)
    # S = dequantize(Q_block_fp8 @ K_block_fp8.T, scale_q * scale_k)
    # Softmax 计算在 FP32
    # attn_weights = softmax(S / sqrt(d))
    # V 用 FP8
    # O = dequantize(attn_weights @ V_block_fp8, scale_v)
    pass
```


### 4.3 Flash Attention 各版本对比


| 特性 | V1 | V2 | V3 |
| --- | --- | --- | --- |
| 发布时间 | 2022.05 | 2023.07 | 2024.07 |
| 核心改进 | IO-aware tiling | 并行化优化 | 硬件级优化 |
| 目标硬件 | A100 | A100 | H100 |
| 精度支持 | FP16/BF16 | FP16/BF16 | FP16/BF16/FP8 |
| A100 吞吐 | ~270 TFLOPS | ~330 TFLOPS | - |
| H100 吞吐 | - | ~350 TFLOPS | ~740 TFLOPS |
| 内存节省 | 5-20x | 5-20x | 5-20x |


## 5. 实际性能对比


### 5.1 吞吐量对比（A100 80GB）


| 序列长度 | 标准注意力 | FlashAttention-1 | FlashAttention-2 |
| --- | --- | --- | --- |
| 512 | ~120 TFLOPS | ~115 TFLOPS | ~150 TFLOPS |
| 1024 | ~150 TFLOPS | ~200 TFLOPS | ~250 TFLOPS |
| 2048 | ~160 TFLOPS | ~250 TFLOPS | ~300 TFLOPS |
| 4096 | ~165 TFLOPS | ~270 TFLOPS | ~330 TFLOPS |
| 8192 | OOM | ~280 TFLOPS | ~340 TFLOPS |


> **Warning:** #### 注意
>
>
> 短序列（n≤512）上标准注意力已经很高效，Flash Attention 优势不明显。Flash Attention 的优势在中长序列（n≥1024）上才充分体现，尤其在避免 OOM 方面更是关键。实际部署中，几乎所有主流框架（PyTorch 2.0+、vLLM、TensorRT-LLM）都已集成 Flash Attention。


### 5.2 内存对比


| 序列长度 | 标准注意力内存 | Flash Attention 内存 | 节省 |
| --- | --- | --- | --- |
| 2048 | ~256 MB | ~16 MB | 16x |
| 4096 | ~1 GB | ~32 MB | 32x |
| 8192 | ~4 GB (OOM) | ~64 MB | 64x |
| 16384 | OOM | ~128 MB | ∞ |


## 6. 使用方式


```
# 方式1: PyTorch 2.0+ 内置
import torch
from torch.nn.functional import scaled_dot_product_attention

# PyTorch 2.0+ 会自动选择 Flash Attention 后端
output = scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)

# 方式2: 直接使用 flash-attn 库
# pip install flash-attn --no-build-isolation
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)

# 方式3: 通过 transformers 库
from transformers import AutoModel
# 默认启用 Flash Attention 2
model = AutoModel.from_pretrained("model", attn_implementation="flash_attention_2")

# 检查是否使用了 Flash Attention
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # True 表示已启用
```


## 7. 总结


#### 关键要点


1. Flash Attention 的核心是
   **IO-aware**
   ：它关注的不是减少 FLOPs，而是减少 HBM 访问
2. **Tiling + Online Softmax**
   是关键技术：将计算限制在 SRAM 内，避免中间矩阵落盘到 HBM
3. 输出与标准注意力
   **数学等价**
   （只是浮点误差不同），不影响模型质量
4. V1 奠定了算法基础，V2 优化了并行化，V3 针对 H100 做了硬件级深度优化
5. 几乎所有现代 LLM 训练和推理框架都已默认使用 Flash Attention
6. Flash Attention 是
   **精确算法**
   ，不是近似算法，这是它相比线性注意力等方法的关键优势

Flash Attention - IO-aware算法、tiling策略、online softmax、V1/V2/V3对比、性能分析完整笔记


<!-- Converted from: 02_Flash_Attention.html -->
