# 01_PyTorch 张量操作与广播机制 (Broadcasting)

## 核心概念

- **张量 (Tensor)**：PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速和自动微分。张量是多维数组的泛化，标量是 0 维，向量是 1 维，矩阵是 2 维。
- **广播机制 (Broadcasting)**：当两个张量形状不匹配时，PyTorch 会自动将较小的张量扩展为与较大张量相同的形状，使得逐元素操作能够执行。广播避免了显式复制数据带来的内存浪费。
- **形状规则**：两个张量从尾部维度开始对齐，如果某个维度大小相同或其中一个为 1，则可广播；否则报错。结果形状取每个维度上的最大值。
- **原地操作 (In-place)**：以 `_` 结尾的操作（如 `add_`, `copy_`）会直接修改张量内容而不创建新副本，节省内存但会破坏计算图历史，在 Autograd 中需谨慎使用。
- **视图 (View) vs 拷贝 (Copy)**：`view()`、`reshape()` 等方法返回的新张量可能与原始张量共享内存（视图），而 `clone()` 返回独立的内存副本。修改视图可能影响原始张量。
- **设备与数据类型管理**：通过 `.to(device)` 在不同设备（CPU/CUDA）间迁移张量，通过 `.to(dtype)` 转换精度（float32/float16/bfloat16）。显式管理设备和类型是工程化部署的关键习惯。

## 数学推导

广播机制的数学本质是维度扩展与重复。假设有两个张量 $A \in \mathbb{R}^{m \times 1 \times n}$ 和 $B \in \mathbb{R}^{1 \times k \times n}$，计算 $C = A + B$ 时：

- 对齐形状：$A.shape = (m, 1, n)$，$B.shape = (1, k, n)$
- 比较每个维度：第 0 维 (m vs 1) --> 广播 B，第 1 维 (1 vs k) --> 广播 A，第 2 维 (n vs n) --> 匹配
- 结果形状：$C \in \mathbb{R}^{m \times k \times n}$

广播不实际复制内存，而是通过 stride 技巧实现"虚扩展"：

$$
\text{stride}_{\text{广播后}} = \begin{cases}
0 & \text{如果该维度原始大小为 1} \\
\text{原始 stride} & \text{否则}
\end{cases}
$$

这使得广播在内存和计算上都极其高效。

## 直观理解

- 广播可以类比为"自动补齐"：想象你有两个不同尺寸的网格，Python 会自动把小网格拉伸到大网格的尺寸，然后执行逐元素运算。关键约束是：两个张量的维度必须兼容（相等或其中一个为 1）。
- **最佳实践**：在编写自定义层时，尽量利用广播而非手动 `expand` 或 `repeat`。广播在 GPU 上的开销几乎为零，而显式扩展会分配额外内存。
- **常见陷阱**：`(3,)` 和 `(3,1)` 形状不同——前者是一维向量，后者是二维列矩阵。使用 `unsqueeze()` 或 `reshape()` 显式管理维度可以避免意外广播。
- **经验法则**：当遇到形状不匹配错误时，从最后一个维度往前检查；在调试时可创建 `(1,)` 或 `(1,1)` 的占位维度来手动控制广播方向。

## 代码示例

```python
import torch

# 基本广播：形状 (3,1) + (1,4) -> (3,4)
a = torch.tensor([[1], [2], [3]])      # (3, 1)
b = torch.tensor([[10, 20, 30, 40]])   # (1, 4)
c = a + b                               # (3, 4)
print(c)
# 输出:
# tensor([[11, 21, 31, 41],
#         [12, 22, 32, 42],
#         [13, 23, 33, 43]])

# 标量广播：形状 (2,3) + () -> (2,3)
x = torch.randn(2, 3)
y = x + 5.0  # 标量被广播为 (2,3)

# 显式控制：unsqueeze 与 expand
scores = torch.randn(4, 8)          # 4个样本，8个类别
weights = torch.tensor([0.5, 1.0])  # 2个权重，需要广播
# 错误写法：scores * weights 会报错
# 正确写法：先 unsqueeze 对齐维度
weights_aligned = weights.unsqueeze(1)  # (2,1)
scores_aligned = scores.reshape(4, 4, 2)  # 假设需要按最后一维加权
# 实际场景中更常见的做法：
batch_norm_bias = torch.randn(64)  # (64,)
feature_map = torch.randn(16, 64, 7, 7)  # (N, C, H, W)
result = feature_map + batch_norm_bias[None, :, None, None]  # 广播到 (16,64,7,7)

# 设备与类型管理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1, 2, 3], device=device, dtype=torch.float32)
y = x.to(torch.bfloat16)  # BF16 转换，AMP 中常用

# 性能对比：广播 vs expand
import time
large = torch.randn(10000, 1, 1000)
small = torch.randn(1, 500, 1)
t0 = time.time()
_ = large + small  # 广播，零拷贝
print(f"Broadcast: {time.time()-t0:.4f}s")
t0 = time.time()
_ = large.expand(-1, 500, -1) + small.expand(10000, -1, 1000)
print(f"Expand first: {time.time()-t0:.4f}s")  # 显著更慢
```

## 深度学习关联

- **Batch Normalization 推理阶段**：BN 层的 running_mean 和 running_var 形状为 (C,)，在推理时通过与输入特征图 (N, C, H, W) 广播实现归一化，避免了逐通道循环。
- **注意力机制中的掩码 (Masking)**：Transformer 中注意力分数矩阵与注意力掩码的广播加法是广播机制最经典的应用之一。通过 `(B, H, N, N) + (B, 1, 1, N)` 实现因果掩码，无需显式扩展掩码矩阵。
- **损失函数中的类别权重**：在交叉熵损失中，类别权重向量 `(C,)` 与 `(B,)` 标签的广播组合实现了对每个样本的加权，这在类别不平衡训练中是标准做法。
- **现代 MLOps 实践**：在模型导出为 ONNX/TensorRT 时，广播操作会被保留为底层算子的属性，理解广播规则有助于排查导出后的形状不兼容问题；在分布式训练中，广播也是 `all_reduce` 等集体通信操作的核心数据流模式。
