# 9_Optuna 框架实战

## 1. Optuna 简介

**Optuna** 是 Preferred Networks 开发的超参数优化框架，以 **Trial API** 为核心设计理念。

### 1.1 核心特点

- **Define-by-run**：搜索空间在运行时动态定义
- **高效采样器**：TPE、CMA-ES、Grid/Random
- **Pruner**：支持早停（Hyperband、Median等）
- **可视化**：内置丰富的可视化功能
- **分布式**：支持多进程/多节点并行

## 2. 基本使用

### 2.1 安装

```bash
pip install optuna
```

### 2.2 第一个例子

```python
import optuna

def objective(trial):
    """目标函数：Optuna会自动调用此函数"""
    # 定义搜索空间
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_int('y', -10, 10)
    
    # 评估
    return (x - 2) ** 2 + (y + 3) ** 2

# 创建study并优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 结果
print(f"最优值: {study.best_value}")
print(f"最优参数: {study.best_params}")
```

## 3. Trial API 详解

### 3.1 搜索空间定义

```python
def objective(trial):
    # 连续参数
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # 整数参数
    n_layers = trial.suggest_int('n_layers', 2, 8)
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)
    
    # 类别参数
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'AdamW'])
    activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'silu'])
    
    # 条件参数
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
    
    return accuracy
```

### 3.2 完整PyTorch示例

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def objective(trial):
    # ====== 搜索超参数 ======
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    
    # ====== 构建模型 ======
    layers = []
    in_size = 784
    for i in range(n_layers):
        out_size = hidden_size
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_size = out_size
    layers.append(nn.Linear(in_size, 10))
    model = nn.Sequential(*layers).to(device)
    
    # ====== 优化器 ======
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # ====== 数据加载 ======
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # ====== 训练 ======
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        # ====== 验证 ======
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x).argmax(dim=1)
                correct += (pred == batch_y).sum().item()
        accuracy = correct / len(val_dataset)
        
        # ====== 报告中间结果（用于Pruning） ======
        trial.report(accuracy, epoch)
        
        # ====== 检查是否应该Prune ======
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return accuracy
```

## 4. 采样器 (Sampler)

### 4.1 内置采样器

| 采样器 | 类 | 特点 |
|--------|-----|------|
| TPE (默认) | `TPESampler` | 贝叶斯优化 |
| CMA-ES | `CmaEsSampler` | 连续空间优化 |
| Grid | `GridSampler` | 穷举搜索 |
| Random | `RandomSampler` | 随机采样 |
| GP | `GPSampler` | 高斯过程 |

```python
# 使用TPE采样器
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    direction='maximize'
)

# 使用CMA-ES（适合连续参数）
study = optuna.create_study(
    sampler=optuna.samplers.CmaEsSampler(),
    direction='maximize'
)

# 使用Grid Search
search_space = {
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [32, 64, 128],
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction='maximize'
)
```

## 5. Pruner（剪枝器）

### 5.1 早停机制

Pruner 决定何时停止不好的试验：

| Pruner | 原理 | 推荐度 |
|--------|------|--------|
| MedianPruner | 中位数剪枝 | 推荐 |
| HyperbandPruner | Hyperband | 最推荐 |
| PercentilePruner | 百分位剪枝 | 灵活 |
| PatientPruner | 容忍N步 | 稳健 |

```python
# Hyperband Pruner
study = optuna.create_study(
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,      # 最少训练epoch数
        max_resource=100,    # 最多训练epoch数
        reduction_factor=3,
    )
)

# Median Pruner
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # 前5个trial不剪枝
        n_warmup_steps=5,      # 每个trial前5个epoch不剪枝
    )
)
```

## 6. 可视化

```python
import optuna.visualization as vis

# 优化历史
fig = vis.plot_optimization_history(study)
fig.show()

# 参数重要性
fig = vis.plot_param_importances(study)
fig.show()

# 参数关系
fig = vis.plot_parallel_coordinate(study)
fig.show()

# 切片图
fig = vis.plot_slice(study, params=['lr', 'hidden_size'])
fig.show()

# 等高线图
fig = vis.plot_contour(study, params=['lr', 'dropout'])
fig.show()
```

## 7. 高级功能

### 7.1 多目标优化

```python
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # 返回多个目标
    accuracy = train_model(lr)
    latency = measure_latency(model)
    return accuracy, latency

study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(objective, n_trials=100)

# Pareto前沿
print(f"Pareto最优试验数: {len(study.best_trials)}")
```

### 7.2 分布式优化

```bash
# 终端1
optuna create-study --study-name "distributed" --storage "sqlite:///example.db"

# 终端2
python train.py --storage "sqlite:///example.db" --study-name "distributed"

# 终端3
python train.py --storage "sqlite:///example.db" --study-name "distributed"
```

```python
# 代码中使用
study = optuna.create_study(
    study_name='distributed',
    storage='sqlite:///example.db',
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)
```

### 7.3 回调函数

```python
def print_best_trial(study, trial):
    """每完成一个trial打印最优结果"""
    print(f"Trial {trial.number} finished. Best: {study.best_value:.4f}")

study.optimize(objective, n_trials=100, callbacks=[print_best_trial])
```

### 7.4 手动管理Trial

```python
study = optuna.create_study()

# 手动创建trial
trial = study.ask()
params = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

# 评估
score = train_and_evaluate(params)

# 报告结果
study.tell(trial, score)
```

## 8. 实战模板

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch

def optimize_model(train_dataset, val_dataset, n_trials=100):
    """完整超参数优化流程"""
    
    def objective(trial):
        # 搜索空间
        config = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
            'n_layers': trial.suggest_int('n_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }
        
        # 构建、训练、评估
        model = build_model(config)
        for epoch in range(20):
            train_one_epoch(model, train_dataset, config)
            val_acc = evaluate(model, val_dataset)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return val_acc
    
    # 创建study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    
    # 优化
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    
    print(f"最优精度: {study.best_value:.4f}")
    print(f"最优配置: {study.best_params}")
    
    return study
```

---

**关键要点**：
1. Optuna 的 Trial API 让搜索空间定义自然融入代码
2. TPE 采样器是默认选择，适合大多数场景
3. HyperbandPruner 通过早停显著加速搜索
4. 多目标优化和分布式支持让 Optuna 适应生产环境
