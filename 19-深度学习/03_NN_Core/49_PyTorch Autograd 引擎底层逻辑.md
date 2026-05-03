# 49_PyTorch Autograd 引擎底层逻辑

## 核心概念

- **Autograd 概述**：PyTorch 的 Autograd 引擎是实现自动微分的核心系统。它通过追踪对需要梯度的张量（`requires_grad=True`）的所有操作，动态构建计算图，然后自动计算梯度。
- **计算图的动态构建**：在每次前向传播时，Autograd 动态构建一个由 `Function` 节点组成的有向无环图（DAG）。每个 `Function` 节点记录了前向计算和反向梯度计算方法。图结构随每次执行而变化，不需要预先定义。
- **Tensor 对象的核心属性**：每个 Tensor 包含 `data`（实际数值）、`grad`（梯度）、`grad_fn`（梯度函数，指向创建该 Tensor 的 Function）、`requires_grad`（是否需要梯度）等属性。`grad_fn` 是连接计算图的关键。
- **反向传播的执行**：`loss.backward()` 触发反向传播。Autograd 从 `loss.grad_fn` 开始，按照拓扑排序逆序遍历计算图，依次调用每个 `Function` 的 `backward()` 方法，将梯度传播到叶节点。

## 核心机制

**Tensor 与 Function**：

每个 Tensor 的 `grad_fn` 属性指向一个 `Function` 对象。叶节点（用户创建的 Tensor）的 `grad_fn = None`。

例如：`z = torch.matmul(x, W)` 会创建一个 `MmBackward` 的 Function 对象，记录了前向传播的输入和输出。

**Function 对象的生命周期**：

```python
# 计算图构建
a = torch.tensor([1.0], requires_grad=True)  # grad_fn=None
b = a * 2                                      # grad_fn=<MulBackward0>
c = b + 1                                      # grad_fn=<AddBackward0>
d = c ** 2                                     # grad_fn=<PowBackward0>

# 反向传播
d.backward()

# 计算图被释放（默认情况下）
```

**梯度累积**：

如果多次调用 `backward()`，梯度会在叶节点累积（不是替换）：

```python
for i in range(3):
    y = (x * i).sum()
    y.backward()
# x.grad = 0 + 1 + 2 = 3 (累积)
```

因此在每个训练步前需要调用 `optimizer.zero_grad()` 清空梯度。

**Hook 机制**：

Autograd 提供了 `register_hook` 机制，允许在梯度传播过程中插入自定义操作：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2

def print_grad(grad):
    print(f"梯度: {grad}")
    return grad

y.register_hook(print_grad)
y.sum().backward()
```

## 直观理解

Autograd 可以理解为"记账系统"：
- 每次张量操作就是一笔交易
- `Function` 节点就是交易记录（记录了输入输出和如何计算反向）
- 计算图就是"账本"（记录了所有交易的关系）
- `backward()` 就是"查账"（从最终结果开始，追溯每笔交易的影响）

从工厂流水线的视角看：前向传播是"组装流程"——原料（输入张量）经过各道工序（Function）变成成品（损失值）。每道工序都记录了"如何将输入组装成输出"和"如果出了问题如何追责"。反向传播就是"追责流程"——从成品出发，沿流水线反向追溯，确定每个工序的改进方向（梯度）。

Autograd 的"动态性"体现在：每次前向传播都重新建立"账本"。这意味着如果你的代码中有 `if` 条件或 `for` 循环，计算图会根据实际执行路径自动调整。

## 代码示例

```python
import torch
import torch.nn as nn

# === 计算图构建基础 ===
print("=== 计算图构建 ===")
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 4.0], requires_grad=False)

c = a * b + a  # c = a * b + a
d = c.sum()

print(f"a.grad_fn: {a.grad_fn}")          # None (叶节点)
print(f"c.grad_fn: {c.grad_fn}")          # <AddBackward0>
print(f"d.grad_fn: {d.grad_fn}")          # <SumBackward0>

# 反向传播
d.backward()
print(f"a.grad: {a.grad}")  # = b + 1 = [2.0, 5.0]

# === grad_fn 链式追溯 ===
print("\n=== grad_fn 链追溯 ===")
x = torch.tensor([1.0], requires_grad=True)
y = x ** 3 + 2 * x + 1
z = y ** 2

# 沿 grad_fn 链反向遍历
current = z.grad_fn
print("计算图链:")
while current is not None:
    print(f"  {current.name()} (输入: {current.input_size})")
    current = current.next_functions[0][0] if current.next_functions else None

# === 自动微分推导 ===
# z = y^2, y = x^3 + 2x + 1
# dz/dx = dz/dy * dy/dx = 2y * (3x^2 + 2)
# 在 x=1 时: y = 4, dz/dx = 8 * 5 = 40

x = torch.tensor([1.0], requires_grad=True)
z = (x ** 3 + 2 * x + 1) ** 2
z.backward()
print(f"\n自动梯度: {x.grad.item()} (理论值: 40)")

# === 高阶梯度 ===
print("\n=== 高阶梯度 ===")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 一阶导数
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx at x=2: {grad1.item()} (理论: 12)")

# 二阶导数
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d²y/dx² at x=2: {grad2.item()} (理论: 12)")

# === Hook 机制 ===
print("\n=== 梯度 Hook ===")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()

def backward_hook(grad):
    print(f"  Hook 接收到的梯度: {grad}")
    return grad * 2  # 修改梯度

handle = y.register_hook(backward_hook)
y.backward()
print(f"  x.grad (被 Hook 修改): {x.grad}")
handle.remove()  # 移除 hook

# === 计算图的保存和释放 ===
print("\n=== retain_graph ===")
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
z = x ** 3

# 同时计算两个梯度
y.backward(retain_graph=True)  # 保留计算图
print(f"dy/dx: {x.grad.item()}")
x.grad.zero_()
z.backward()  # 复用计算图
print(f"dz/dx: {x.grad.item()}")

# === 禁用梯度追踪 ===
print("\n=== 禁用梯度追踪 ===")
x = torch.tensor([1.0, 2.0], requires_grad=True)

with torch.no_grad():
    y = x * 2
    print(f"no_grad 下 y.requires_grad: {y.requires_grad}")

# 局部禁用
y = x * 2  # 有梯度
print(f"正常 y.requires_grad: {y.requires_grad}")

# === Autograd 函数自定义 ===
print("\n=== 自定义 Autograd Function ===")
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = CustomReLU.apply(x)
y.sum().backward()
print(f"自定义 ReLU 梯度: {x.grad}")
```

## 深度学习关联

- **深度学习的基石**：Autograd 引擎是 PyTorch 的核心竞争力之一。其"动态图 + 自动微分"的设计使得研究者可以用接近数学公式的方式编写网络代码，同时自动获取梯度。这大大降低了深度学习的开发门槛。
- **性能优化策略**：PyTorch Autograd 通过多种策略优化性能：延迟执行（lazy execution）、中间缓冲区复用、操作融合（fused kernels）。`torch.compile` 进一步通过图捕获和编译优化 Autograd 的性能。
- **分布式 Autograd**：PyTorch 的分布式 Autograd（`torch.distributed.autograd`）扩展了 Autograd 到分布式场景。在模型并行和流水线并行中，梯度需要在不同设备之间传播，分布式 Autograd 管理跨设备的计算图构建和反向传播。
