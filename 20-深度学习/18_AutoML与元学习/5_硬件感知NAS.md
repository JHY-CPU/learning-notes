# 5_硬件感知 NAS

## 1. 为什么需要硬件感知NAS

模型的**理论计算量 (FLOPs)** 并不等于**实际推理延迟 (Latency)**。在移动端部署时，需要直接优化延迟，而非FLOPs。

### 1.1 FLOPs vs 延迟

| 模型 | FLOPs | GPU延迟 | CPU延迟 |
|------|-------|---------|---------|
| MobileNetV2 | 300M | 6ms | 75ms |
| ShuffleNetV2 | 296M | 8ms | 62ms |
| EfficientNet-B0 | 390M | 5ms | 52ms |

相同FLOPs下，实际延迟差异巨大！原因包括：
- 内存访问模式
- 并行度
- 硬件缓存命中率
- 算子融合能力

## 2. 延迟约束的NAS

### 2.1 目标函数

$$\min_\alpha \mathcal{L}_{val}(\alpha) \quad \text{s.t.} \quad \text{Latency}(\alpha) \leq T_{budget}$$

### 2.2 延迟查找表 (Lookup Table)

预先测量每个候选操作在目标硬件上的延迟：

```python
class LatencyLookupTable:
    """延迟查找表"""
    def __init__(self, hardware='mobile_cpu'):
        self.hardware = hardware
        self.table = self._build_table()
    
    def _build_table(self):
        """测量所有候选操作的延迟"""
        table = {}
        candidate_ops = [
            ('conv3x3', 32, 64),
            ('conv3x3', 64, 128),
            ('conv5x5', 32, 64),
            ('sep_conv_3x3', 32, 64),
            ('dil_conv_3x3', 32, 64),
            ('max_pool_3x3', 32, 32),
            ('skip_connect', 32, 32),
            # ... 更多组合
        ]
        
        for op_name, c_in, c_out in candidate_ops:
            key = f"{op_name}_{c_in}_{c_out}"
            model = build_op(op_name, c_in, c_out)
            latency = self._measure_latency(model)
            table[key] = latency
        
        return table
    
    def _measure_latency(self, model, num_runs=100, warmup=10):
        """在目标硬件上测量延迟"""
        import time
        
        device = torch.device('cpu')  # 或 'cuda'
        model = model.to(device).eval()
        dummy_input = torch.randn(1, model.in_channels, 224, 224).to(device)
        
        # Warmup
        for _ in range(warmup):
            model(dummy_input)
        
        # 测量
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append(time.perf_counter() - start)
        
        return np.median(times) * 1000  # ms
    
    def estimate_latency(self, architecture):
        """估计整个架构的延迟"""
        total_latency = 0
        for layer_key, op_name in architecture.items():
            lookup_key = f"{op_name}_{layer_key}"
            total_latency += self.table.get(lookup_key, 0)
        return total_latency
```

## 3. MnasNet

### 3.1 多目标优化

**MnasNet (Tan et al., CVPR 2019)** 同时优化精度和延迟：

$$\max_{\alpha} \text{ACC}(\alpha) \times \left[\frac{\text{Latency}(\alpha)}{T}\right]^\beta$$

其中 $T$ 是目标延迟，$\beta$ 是平衡因子。

```python
def mnasnet_reward(accuracy, latency, target_latency=33, beta=-0.07):
    """MnasNet奖励函数"""
    latency_penalty = (latency / target_latency) ** beta
    return accuracy * latency_penalty
```

### 3.2 搜索空间

MnasNet 的搜索空间包括：
- 每层的操作类型（MBConv核大小、扩展比）
- 每层的通道数
- 是否使用SE模块
- Skip连接模式

### 3.3 搜索结果

MnasNet-A1: 75.2% ImageNet Top-1, 78ms Pixel Phone延迟

## 4. FBNet

### 4.1 可微分硬件感知搜索

**FBNet (Wu et al., CVPR 2019)** 在 DARTS 的基础上加入延迟正则化：

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \text{Latency}(\alpha)$$

```python
def fbnet_loss(pred, target, alpha, latency_table, lambda_lat=1e-2):
    """FBNet损失：交叉熵 + 延迟正则化"""
    ce_loss = F.cross_entropy(pred, target)
    
    # 延迟估计
    latency = latency_table.estimate_latency(alpha)
    
    return ce_loss + lambda_lat * latency
```

## 5. Once-for-All (OFA)

### 5.1 核心思想

**OFA (Cai et al., ICLR 2020)** 训练一个支持多种配置的超网络，部署时为每个硬件选择最优子网络：

```
训练: 一个OFA超网络支持：
  - 不同的核大小 (3, 5, 7)
  - 不同的深度 (2-6层)
  - 不同的宽度 (0.25x ~ 1.0x)
    ↓
部署: 对每个目标硬件，
  从超网络中提取最优子网络
```

### 5.2 渐进式收缩训练

```python
def ofa_training(supernet, dataset):
    """OFA渐进式收缩训练"""
    
    # 阶段1: 训练最大核大小 (7x7)
    train_kernel(supernet, kernel_size=7, epochs=25)
    
    # 阶段2: 同时训练核大小 5 和 7
    for epoch in range(25):
        ks = random.choice([5, 7])
        train_step(supernet, kernel_size=ks)
    
    # 阶段3: 同时训练核大小 3, 5, 7
    for epoch in range(25):
        ks = random.choice([3, 5, 7])
        train_step(supernet, kernel_size=ks)
    
    # 阶段4: 同时训练不同深度
    for epoch in range(25):
        depth = random.randint(2, 6)
        ks = random.choice([3, 5, 7])
        train_step(supernet, depth=depth, kernel_size=ks)
    
    # 阶段5: 同时训练不同宽度
    for epoch in range(25):
        depth = random.randint(2, 6)
        ks = random.choice([3, 5, 7])
        width_mult = random.choice([0.25, 0.5, 0.75, 1.0])
        train_step(supernet, depth=depth, kernel_size=ks, width_mult=width_mult)
```

### 5.3 硬件部署

```python
def ofa_deploy(supernet, target_hardware, num_candidates=1000):
    """为特定硬件选择最优子网络"""
    best_acc = 0
    best_arch = None
    
    for _ in range(num_candidates):
        # 采样子网络配置
        config = {
            'depth': random.randint(2, 6),
            'kernel': random.choice([3, 5, 7]),
            'width': random.choice([0.25, 0.5, 0.75, 1.0]),
        }
        
        # 提取子网络
        subnet = supernet.extract_subnet(config)
        
        # 评估精度
        acc = evaluate(subnet, val_set)
        
        # 测量延迟
        latency = measure_latency(subnet, target_hardware)
        
        # 多目标评估
        if latency <= LATENCY_BUDGET and acc > best_acc:
            best_acc = acc
            best_arch = config
    
    return best_arch
```

## 6. 方法对比

| 方法 | 搜索维度 | 硬件信息 | 延迟优化 |
|------|----------|----------|----------|
| MnasNet | 操作+通道 | 延迟测量 | 奖励函数 |
| FBNet | 操作+通道 | 延迟查找表 | 正则化 |
| OFA | 核大小+深度+宽度 | 延迟测量 | 部署时选择 |
| ProxylessNAS | 操作选择 | 直接延迟 | 隐式优化 |
| HAT | 操作+通道 | 延迟预测器 | 多目标 |

## 7. 实际部署流程

```
1. 定义搜索空间（针对目标硬件）
    ↓
2. 训练超网络 / 搜索
    ↓
3. 在目标硬件上测量延迟
    ↓
4. 选择满足延迟约束的最优架构
    ↓
5. 提取子网络权重
    ↓
6. 微调/量化
    ↓
7. 部署
```

---

**关键要点**：
1. FLOPs不等于延迟，硬件感知NAS直接优化实际推理速度
2. 延迟查找表是高效估计延迟的关键技术
3. OFA通过训练一个支持多种配置的超网络，实现"一次训练，处处部署"
4. 多目标优化（精度 vs 延迟）是硬件感知NAS的核心
