# 深度学习工程与部署

## 一、主流框架对比

| 特性 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| 编程风格 | 动态图（Eager） | 2.x Eager + Graph | 函数式 |
| 调试难度 | 低（Python原生） | 中 | 中 |
| 部署生态 | TorchServe, ONNX | TF Serving, TFLite | — |
| 分布式训练 | DDP, FSDP | tf.distribute | pjit |
| 研究热度 | 最高（学术界主流） | 高（工业界） | 增长中 |

### PyTorch核心API

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 训练循环
model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for x, y in dataloader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 二、训练技巧

### 2.1 学习率调度

- **StepLR**：每N个epoch衰减
- **CosineAnnealing**：余弦退火
- **Warmup + Decay**：先升后降（Transformer标配）
- **OneCycleLR**：一个周期内完成学习率变化

### 2.2 混合精度训练

使用FP16减少显存占用，加速训练：
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2.3 梯度累积

小显存模拟大batch：
```python
for i, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 三、分布式训练

### 3.1 数据并行（DDP）

每个GPU持有完整模型副本，数据分片，梯度同步。

### 3.2 模型并行

- **张量并行**：将单个层切分到多个GPU
- **流水线并行**：不同层放在不同GPU
- **FSDP**：完全分片数据并行（ZeRO）

### 3.3 DeepSpeed

微软的分布式训练框架：
- ZeRO-1/2/3：不同程度的优化器状态分片
- 混合精度、梯度检查点
- 支持万亿参数模型训练

---

## 四、模型部署

### 4.1 模型优化

- **量化**：FP32 → INT8/INT4（QAT、PTQ）
- **剪枝**：移除不重要的权重
- **知识蒸馏**：大模型指导小模型
- **ONNX**：跨框架模型格式

### 4.2 推理引擎

| 工具 | 特点 |
|------|------|
| TensorRT | NVIDIA GPU优化 |
| ONNX Runtime | 跨平台 |
| vLLM | LLM推理优化（PagedAttention） |
| llama.cpp | CPU/边缘设备LLM推理 |
| TFLite | 移动端/嵌入式 |
| TorchServe | PyTorch模型服务 |

### 4.3 大模型推理优化

- **KV Cache**：缓存注意力的Key/Value，避免重复计算
- **Flash Attention**：IO感知的注意力计算
- **PagedAttention**：虚拟内存式KV Cache管理
- **投机解码**：小模型草稿 + 大模型验证
- **连续批处理**：动态合并请求
