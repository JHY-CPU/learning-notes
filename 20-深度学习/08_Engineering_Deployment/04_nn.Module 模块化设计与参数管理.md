# 04_nn.Module 模块化设计与参数管理

## 核心概念

- **nn.Module 基类**：PyTorch 中所有神经网络模块的基类。通过继承此类并实现 `forward` 方法定义网络结构，自动注册子模块和参数，支持 `to(device)`、`train()/eval()`、`parameters()` 等统一管理方法。
- **参数注册与状态管理**：`nn.Parameter` 包装的张量会被自动注册为模块的可训练参数，通过 `module.parameters()` 返回。缓冲区 (buffer) 通过 `register_buffer()` 注册（如 BN 的 running_mean），不属于参数但会随 `to(device)` 迁移。
- **前向传播 (forward)**：`__call__` 内部调用 `forward` 方法，同时触发 hooks 注册。建议始终重写 `forward` 而非直接调用，多态设计使得子模块组合灵活。
- **模块组合与嵌套**：Module 可以递归包含子 Module，PyTorch 自动遍历整个模块树进行参数收集和设备迁移。这一特性使得构建复杂网络非常简洁。
- **参数冻结 (Freeze)**：设置 `requires_grad=False` 可冻结参数，在微调和迁移学习中广泛使用。冻结部分通过 `filter(lambda p: p.requires_grad, model.parameters())` 传给优化器。
- **Hook 机制**：通过 `register_forward_hook`、`register_backward_hook` 等钩子，可以在不修改 forward 代码的情况下插入调试、特征提取、梯度裁剪等逻辑。

## 数学推导

Module 的参数空间可以视为一个结构化的 n 维流形。对于一个有 $L$ 层的网络，参数集为：

$$
\Theta = \{\theta_1, \theta_2, ..., \theta_L\}, \quad \theta_l \in \mathbb{R}^{d_l}
$$

前向传播的计算图是一个有向无环图 (DAG)：

$$
x_{l} = f_l(x_{l-1}; \theta_l), \quad l = 1,...,L
$$

其中 $f_l$ 可以是线性变换、卷积、归一化等操作。梯度反向传播遵循链式法则：

$$
\frac{\partial \mathcal{L}}{\partial \theta_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{k=l+1}^{L} \frac{\partial x_k}{\partial x_{k-1}} \cdot \frac{\partial x_l}{\partial \theta_l}
$$

Module 的模块化设计正是对这种链式结构的直接编码，每个子模块封装了自己的参数和 forward 逻辑。

## 直观理解

- **nn.Module = 乐高积木**：每个 Module 就像一块乐高积木，有自己的形状（输入输出维度）和连接方式。你可以用 `Sequential` 将它们简单拼接，也可以用自定义 `forward` 实现复杂的跳连和分支结构。
- **最佳实践**：始终在 `__init__` 中定义所有子模块，在 `forward` 中实现动态路由逻辑。不要动态创建新的 `nn.Conv2d` 等模块，这会绕过参数管理。
- **常见陷阱**：忘记调用父类 `super().__init__()` 会导致参数注册失败；将普通 `torch.Tensor` 而非 `nn.Parameter` 放入 Module 会导致参数不被优化器识别。
- **经验法则**：当模块变得复杂时（超过 100 行），考虑拆分为多个子 Module；使用 `apply()` 方法可以递归地对所有子模块应用初始化函数。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基础模块定义
class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        return self.dropout(x)

# 2. 带残差连接的模块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# 3. 参数冻结与选择性优化
class FineTuneModel(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.out_dim, num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.classifier(x)

# 4. Hook 示例
activation = {}
def get_hook(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

model = ResidualBlock(64)
model.conv1.register_forward_hook(get_hook("conv1_out"))
dummy = torch.randn(4, 64, 32, 32)
_ = model(dummy)
print(f"conv1 output shape: {activation['conv1_out'].shape}")

# 5. 参数初始化
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)

# 6. 参数统计
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

## 深度学习关联

- **大规模模型的分片 (Sharding)**：当模型参数量超过单 GPU 显存时，nn.Module 的参数管理需要和 FSDP/Deepspeed 的分片策略结合。每个子 Module 的参数量决定了分片粒度——选择多大尺寸的模块作为分片单元直接影响通信效率和内存平衡。
- **推理优化与模型导出**：在生产环境中，经过训练的 nn.Module 需要通过 `torch.jit.trace` 或 `torch.onnx.export` 转换为静态图。这要求 forward 中不包含动态控制流（if/for 依赖输入），否则 trace 会失败。
- **实验管理中的元数据**：在 MLflow 中记录模型时，通常递归遍历 `named_parameters()` 和 `named_buffers()` 以获取所有可训练参数的总数、每层维度等元数据，用于后续的模型比较和回归分析。
