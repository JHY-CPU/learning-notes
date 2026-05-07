# 1_NAS 神经架构搜索基础

## 1. NAS 概述

**神经架构搜索 (Neural Architecture Search, NAS)** 自动设计神经网络的架构，是AutoML最核心的子领域。

### 1.1 NAS 的三个核心要素

```
搜索空间 (Search Space)
    ↕
搜索策略 (Search Strategy)
    ↕
性能评估 (Performance Estimation)
```

| 要素 | 问题 | 方法 |
|------|------|------|
| 搜索空间 | 搜索什么？ | 链式/多分支/Cell-based |
| 搜索策略 | 如何搜索？ | RL/进化/可微分 |
| 性能评估 | 如何评估？ | 训练到收敛/早停/权重共享 |

## 2. 搜索空间

### 2.1 宏搜索空间 (Macro Search Space)

直接搜索整个网络的架构：

```python
macro_search_space = {
    'num_layers': range(5, 20),
    'layer_config': [
        {
            'type': ['conv3x3', 'conv5x5', 'conv7x7', 'maxpool3x3', 'identity'],
            'filters': [32, 64, 128, 256],
            'activation': ['relu', 'gelu'],
            'use_bn': [True, False],
        }
        for _ in range(20)  # 每层的配置
    ]
}
```

**问题**：搜索空间巨大，组合爆炸。

### 2.2 Cell-based 搜索空间

**核心思想**：只搜索一个**Cell（单元）**的结构，然后堆叠多个Cell。

```
Cell = N个节点的DAG（有向无环图）
    节点 = 特征张量
    边 = 操作（卷积、池化等）
```

```
完整网络:
    Stem → Cell × N1 → Reduction Cell → Cell × N2 → Reduction Cell → Cell × N3 → 分类头
```

```python
class Cell(nn.Module):
    """NAS搜索的Cell结构"""
    def __init__(self, num_nodes=4, candidate_ops=None):
        super().__init__()
        self.num_nodes = num_nodes
        if candidate_ops is None:
            candidate_ops = [
                'sep_conv_3x3', 'sep_conv_5x5',
                'dil_conv_3x3', 'dil_conv_5x5',
                'max_pool_3x3', 'avg_pool_3x3',
                'skip_connect', 'zero',
            ]
        
        # 每条边的候选操作
        self.edges = nn.ModuleDict()
        for i in range(num_nodes):
            for j in range(i):
                # 节点j到节点i的边，每条边有多个候选操作
                self.edges[f'{j}_{i}'] = nn.ModuleList([
                    OPS[op](C, stride=1) for op in candidate_ops
                ])
    
    def forward(self, x, weights):
        """
        x: 输入特征
        weights: 每条边每个操作的权重 (来自架构参数)
        """
        nodes = [x]  # 节点0 = 输入
        
        for i in range(1, self.num_nodes):
            node_feat = 0
            for j in range(i):
                edge_key = f'{j}_{i}'
                # 加权求和所有候选操作
                edge_ops = self.edges[edge_key]
                w = weights[edge_key]  # (num_ops,)
                node_feat += sum(
                    w[k] * edge_ops[k](nodes[j]) 
                    for k in range(len(edge_ops))
                )
            nodes.append(node_feat)
        
        return nodes[-1]  # 输出最后一个节点
```

### 2.3 操作定义

```python
OPS = {
    'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
    'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride, 1),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride, 1),
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C),
    'zero': lambda C, stride: Zero(stride),
}

class SepConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            # Depthwise
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in), nn.ReLU(),
            # Pointwise
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out), nn.ReLU(),
        )
    
    def forward(self, x):
        return self.op(x)
```

## 3. 搜索策略

### 3.1 三大范式

| 范式 | 方法 | 代表 | 计算成本 |
|------|------|------|----------|
| 从零训练每个候选 | RL / 进化 | NASNet | 极高 (2000+ GPU天) |
| 权重共享 (One-Shot) | 超网络 | ENAS, DARTS | 低 (1-2 GPU天) |
| 性能预测器 | 代理模型 | BRP-NAS | 中 |

### 3.2 强化学习范式

用一个**控制器RNN**生成架构描述：

```
控制器RNN → 生成架构 → 训练该架构 → 验证精度 → 奖励信号 → 更新控制器
```

### 3.3 进化算法范式

```
初始化种群 (随机架构)
    ↓
评估适应度 (训练+验证精度)
    ↓
选择 + 交叉 + 变异
    ↓
新种群
    ↓
重复
```

```python
def evolutionary_nas(population_size=50, generations=100):
    """进化算法NAS"""
    # 初始化种群
    population = [random_architecture() for _ in range(population_size)]
    
    for gen in range(generations):
        # 评估适应度
        fitness = []
        for arch in population:
            acc = train_and_evaluate(arch, epochs=10)  # 低保真度评估
            fitness.append(acc)
        
        # 选择（锦标赛选择）
        parents = tournament_selection(population, fitness, k=3)
        
        # 交叉
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child = crossover(parents[i], parents[i+1])
                children.append(child)
        
        # 变异
        children = [mutate(child, rate=0.1) for child in children]
        
        # 新种群
        population = select_survivors(population + children, fitness, population_size)
    
    # 返回最优架构
    best_idx = np.argmax(fitness)
    return population[best_idx]
```

## 4. 性能评估

### 4.1 评估挑战

训练一个架构到收敛需要大量计算。NAS需要评估数千个候选，因此需要**快速评估**策略。

### 4.2 评估方法对比

| 方法 | 精度 | 速度 | 说明 |
|------|------|------|------|
| 完整训练 | 最高 | 最慢 | 训练数百个epoch |
| 早停 | 高 | 快 | 训练少部分epoch |
| 学习曲线外推 | 中 | 很快 | 从早期曲线预测 |
| 权重共享 | 低 | 极快 | 所有架构共享超网络权重 |
| 性能预测器 | 中 | 最快 | 训练代理模型预测 |

### 4.3 低保真度评估

```python
def low_fidelity_evaluate(architecture, dataset, budget='small'):
    """低保真度评估"""
    budgets = {
        'tiny': {'epochs': 5, 'subset_ratio': 0.1, 'resolution': 64},
        'small': {'epochs': 10, 'subset_ratio': 0.25, 'resolution': 128},
        'medium': {'epochs': 50, 'subset_ratio': 0.5, 'resolution': 224},
        'full': {'epochs': 200, 'subset_ratio': 1.0, 'resolution': 224},
    }
    config = budgets[budget]
    
    model = build_model(architecture)
    
    # 使用子集
    subset = random_subset(dataset, config['subset_ratio'])
    
    # 降分辨率
    transform = transforms.Resize(config['resolution'])
    
    # 训练
    for epoch in range(config['epochs']):
        train_one_epoch(model, subset, transform)
    
    # 验证
    acc = evaluate(model, val_set)
    return acc
```

## 5. NAS 流程总结

```python
def nas_pipeline(search_space, dataset, num_searches=1000):
    """NAS完整流程"""
    best_arch = None
    best_acc = 0
    
    for i in range(num_searches):
        # 1. 采样架构
        if search_strategy == 'random':
            arch = search_space.sample_random()
        elif search_strategy == 'rl':
            arch = controller.sample_architecture()
        elif search_strategy == 'evolutionary':
            arch = evolve(population)
        
        # 2. 快速评估
        acc = low_fidelity_evaluate(arch, dataset, budget='small')
        
        # 3. 更新最优
        if acc > best_acc:
            best_acc = acc
            best_arch = arch
        
        # 4. 更新搜索策略
        if search_strategy == 'rl':
            controller.update(arch, acc)
    
    # 5. 最优架构完整训练
    final_model = build_model(best_arch)
    final_acc = train_and_evaluate(final_model, dataset, epochs=200)
    
    return best_arch, final_acc
```

## 6. 经典结果

| 方法 | 年份 | 搜索成本 | ImageNet Top-1 |
|------|------|----------|----------------|
| NASNet | 2017 | 2000 GPU天 | 74.0% |
| AmoebaNet | 2018 | 3150 GPU天 | 75.7% |
| ENAS | 2018 | 0.5 GPU天 | 74.3% |
| DARTS | 2019 | 1.5 GPU天 | 73.3% |
| EfficientNet-B0 | 2019 | — | 77.1% |

---

**关键要点**：
1. NAS的三大核心要素：搜索空间、搜索策略、性能评估
2. Cell-based搜索空间大幅缩小搜索范围，是主流设计
3. 权重共享（One-Shot）显著降低搜索成本，但精度有折衷
4. 性能评估是NAS效率的瓶颈，早停和低保真度是常用加速手段
