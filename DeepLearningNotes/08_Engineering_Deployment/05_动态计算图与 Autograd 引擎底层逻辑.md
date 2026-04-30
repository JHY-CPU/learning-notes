# 05_动态计算图与 Autograd 引擎底层逻辑

## 核心概念

- **动态计算图 (Dynamic Computation Graph)**：PyTorch 在每次前向传播时即时构建计算图，图结构由代码执行路径动态决定。这与 TensorFlow 1.x 的静态图不同，允许在 forward 中使用 Python 控制流（if/for/while），极大提升了灵活性。
- **Autograd 引擎**：PyTorch 的自动微分引擎。通过记录所有张量操作构建计算图，然后利用链式法则自动计算梯度。其核心数据结构是 `Function` 对象，每个操作都对应一个 `Function`，连接着输入和输出张量。
- **Tensor 的 grad_fn 属性**：每个参与计算的张量都有一个 `grad_fn` 属性，指向创建该张量的 `Function`（叶子张量的 `grad_fn` 为 None）。这个属性构成了反向传播的入口点。
- **反向传播过程**：调用 `loss.backward()` 时，Autograd 从 loss 张量出发，沿计算图反向遍历，依次调用每个 `Function` 的 `backward` 方法，将梯度累积到叶子张量的 `.grad` 属性中。
- **计算图释放与 retain_graph**：默认情况下，每次 `backward()` 后计算图会被释放以节省内存。若需要多次反向传播（如 GAN 训练），需设置 `retain_graph=True`。
- **requires_grad 与 no_grad**：`requires_grad=True` 的张量会被追踪并参与梯度计算；`torch.no_grad()` 上下文管理器临时禁用梯度追踪，在推理时显著减少内存消耗。

## 数学推导

Autograd 的核心是反向模式自动微分 (Reverse-mode AD)。考虑一个复合函数 $y = f(g(h(x)))$：

前向传播：
$$
a = h(x), \quad b = g(a), \quad y = f(b)
$$

反向传播（链式法则）：
$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial b} \cdot \frac{\partial b}{\partial a} \cdot \frac{\partial a}{\partial x}
$$

对于矩阵乘法 $C = A \cdot B$，其中 $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$：

forward: $C_{ij} = \sum_{t=1}^k A_{it} B_{tj}$

backward（已知 $\frac{\partial L}{\partial C}$）：
$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T \in \mathbb{R}^{m \times k}
$$
$$
\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C} \in \mathbb{R}^{k \times n}
$$

这被称为向量-雅可比积 (VJP, Vector-Jacobian Product)，是 Autograd 引擎的核心计算模式。

## 直观理解

- **计算图 = 烘焙食谱**：前向传播是照食谱一步步操作，Autograd 记录了每一步操作的"逆向配方"。当你需要知道"调整面粉量对成品甜度的影响"时，Autograd 从成品反向追溯每一步，计算出每个原料的影响程度。
- **最佳实践**：每次 backward 前必须调用 `optimizer.zero_grad()` 清空梯度，否则梯度会累积。梯度累积正是利用这一特性模拟更大的 batch size。
- **常见陷阱**：原地操作（如 `x += 1` 或 `x.add_(1)`）修改了张量值但未更新计算图记录，会导致 Autograd 计算梯度时引用错误的数据而报错。
- **经验法则**：推理时始终使用 `torch.inference_mode()`（性能优于 `no_grad`）；在 debug 梯度问题时，检查 `tensor.grad_fn` 是否为 None 即可判断该张量是否为叶子节点。

## 代码示例

```python
import torch

# 1. 计算图基础
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.0, -1.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

z = torch.dot(x, w) + b
loss = z.sigmoid()

print(f"z.grad_fn = {z.grad_fn}")        # AddBackward0
print(f"loss.grad_fn = {loss.grad_fn}")  # SigmoidBackward0
print(f"x.grad_fn = {x.grad_fn}")        # None (叶子节点)

loss.backward()
print(f"grad of x: {x.grad}")
print(f"grad of w: {w.grad}")
print(f"grad of b: {b.grad}")

# 2. retain_graph 多次反向传播
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print(f"First backward: {x.grad}")   # 2.0
y.backward()
print(f"Second backward: {x.grad}")  # 4.0 (累积)

# 3. inference_mode 推理优化
x = torch.randn(100, 64, requires_grad=True)
w = torch.randn(64, 10, requires_grad=True)
with torch.inference_mode():
    out = x @ w
    print(f"推理时 requires_grad = {out.requires_grad}")  # False

# 4. 梯度累积模拟更大 batch
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
accumulation_steps = 4
# for i, (data, target) in enumerate(loader):
#     loss = model(data).sum() / accumulation_steps
#     loss.backward()
#     if (i + 1) % accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()

# 5. 自定义 autograd Function
class MyReLU(torch.autograd.Function):
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

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(f"自定义 ReLU 梯度: {x.grad}")  # [0., 0., 1.]
```

## 深度学习关联

- **梯度检查点 (Activation Checkpointing)**：在训练大型模型（如 GPT、LLaMA）时，中间激活值占用的显存超过参数本身。Gradient Checkpointing 通过在 forward 时丢弃中间激活，在 backward 时重新计算，以时间换空间。自定义 `Function` 的 `save_for_backward` 是实现这一优化的基础。
- **混合精度训练 (AMP) 与 Autograd**：PyTorch 的 `torch.cuda.amp.autocast` 自动在前向传播中选择 FP16/BF16 精度，Autograd 引擎在 backward 时自动匹配对应的精度梯度。理解梯度缩放 (Gradient Scaling) 机制对调试 AMP 训练中的 NaN 损失至关重要。
- **分布式训练中的梯度同步**：在 DDP 中，每个 Autograd backward 完成后自动触发梯度 all-reduce 同步。这一集成是通过在 `backward()` 函数中注册 hook 完成的，理解这一机制有助于诊断分布式训练中的梯度不一致问题。
