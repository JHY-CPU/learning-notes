# Optuna与实践


## 一、Optuna 核心概念


Optuna 是一个开源的超参数优化框架，由 Preferred Networks 开发，支持 Python 原生语法定义搜索空间。


### 1.1 三大核心概念


| 概念 | 说明 | 类比 |
| --- | --- | --- |
| **Study** | 一次超参数优化的完整过程 | 实验项目 |
| **Trial** | 一次超参数组合的评估 | 一次实验 |
| **Objective** | 目标函数，接受trial返回评估指标 | 实验方案 |


### 1.2 搜索空间定义


```
# Optuna搜索空间定义方法
def objective(trial):
    # 整数参数
    n_layers = trial.suggest_int('n_layers', 1, 5)

    # 浮点数参数（均匀分布）
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # 对数均匀

    # 浮点数参数（步进）
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

    # 类别参数
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    # 条件搜索空间
    if optimizer == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
```


## 二、剪枝策略（Pruning）


剪枝是Optuna的重要特性：在trial运行过程中，如果发现当前配置表现明显差于已有结果，提前终止该trial，节省计算资源。


### 2.1 内置剪枝器


| 剪枝器 | 策略 | 适用场景 |
| --- | --- | --- |
| MedianPruner | 当前值低于历史中位数则剪枝 | 通用，默认选择 |
| HyperbandPruner | 多保真度+SuccessiveHalving | 深度学习训练 |
| PercentilePruner | 当前值低于指定百分位则剪枝 | 更激进的剪枝 |
| PatientPruner | 等待指定轮次无改善再剪枝 | 训练波动大的场景 |


### 2.2 剪枝使用方式


```
def objective(trial):
    model = build_model(trial)

    for epoch in range(100):
        train_loss = train_one_epoch(model)
        val_acc = evaluate(model, val_data)

        # 报告中间结果给Optuna
        trial.report(val_acc, epoch)

        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc
```


> **Note:** **剪枝的价值：**
> 假设100个trial，每个训练100个epoch。没有剪枝需要 100 * 100 = 10000 个epoch计算量。使用剪枝后，平均每个trial可能在20-30个epoch被剪枝，总计算量降至 2000-3000 个epoch，加速3-5倍。


## 三、可视化与分析


Optuna内置了丰富的可视化功能，帮助理解优化过程和超参数的重要性。


```
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)

study = optuna.load_study(study_name="my_study",
                           storage="sqlite:///study.db")

# 1. 优化历史：展示目标值随trial的变化
fig1 = plot_optimization_history(study)
fig1.show()

# 2. 超参数重要性：哪些参数对目标值影响最大
fig2 = plot_param_importances(study)
fig2.show()

# 3. 平行坐标图：展示参数组合与目标值的关系
fig3 = plot_parallel_coordinate(study)
fig3.show()

# 4. 切片图：每个参数与目标值的关系
fig4 = plot_slice(study)
fig4.show()

# 5. 等高线图：两个参数之间的交互效应
fig5 = plot_contour(study, params=['learning_rate', 'n_layers'])
fig5.show()
```


## 四、分布式超参数搜索


Optuna支持分布式优化：多个进程/机器同时运行不同的trial，共享同一个study数据库。


### 4.1 使用数据库后端


```
# 进程1（或机器1）
study = optuna.create_study(
    study_name="distributed_hpo",
    storage="mysql://user:pass@host/db",  # 共享数据库
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

# 进程2（或机器2）— 同时运行
study = optuna.create_study(
    study_name="distributed_hpo",
    storage="mysql://user:pass@host/db",
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

# 两个进程共享同一Study，并行探索不同区域
```


### 4.2 支持的存储后端


| 后端 | URI格式 | 适用场景 |
| --- | --- | --- |
| SQLite | sqlite:///study.db | 单机调试 |
| MySQL | mysql://user:pass@host/db | 小规模分布式 |
| PostgreSQL | postgresql://user:pass@host/db | 生产环境 |
| Redis | redis://host:port | 高速分布式 |


## 五、PyTorch + Optuna 完整示例


> **Example:** ### 示例：Optuna优化PyTorch神经网络
>
>
> ```
> import torch
> import torch.nn as nn
> import torch.optim as optim
> from torch.utils.data import DataLoader, TensorDataset
> import optuna
> from sklearn.datasets import make_classification
> from sklearn.model_selection import train_test_split
> from sklearn.preprocessing import StandardScaler
> import numpy as np
>
> # 1. 准备数据
> X, y = make_classification(n_samples=2000, n_features=20,
>                            n_informative=15, random_state=42)
> scaler = StandardScaler()
> X = scaler.fit_transform(X)
>
> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
>                                                      random_state=42)
>
> train_dataset = TensorDataset(torch.FloatTensor(X_train),
>                                torch.LongTensor(y_train))
> val_dataset = TensorDataset(torch.FloatTensor(X_val),
>                              torch.LongTensor(y_val))
>
> # 2. 定义目标函数
> def objective(trial):
>     # 搜索网络结构
>     n_layers = trial.suggest_int('n_layers', 1, 4)
>     layers = []
>     in_features = 20
>
>     for i in range(n_layers):
>         out_features = trial.suggest_int(f'n_units_l{i}', 16, 128)
>         layers.append(nn.Linear(in_features, out_features))
>
>         activation = trial.suggest_categorical(f'activation_l{i}',
>                                                 ['ReLU', 'Tanh', 'GELU'])
>         layers.append(getattr(nn, activation)())
>
>         dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
>         layers.append(nn.Dropout(dropout))
>
>         in_features = out_features
>
>     layers.append(nn.Linear(in_features, 2))
>     model = nn.Sequential(*layers)
>
>     # 搜索优化器参数
>     lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
>     optimizer_name = trial.suggest_categorical('optimizer',
>                                                 ['Adam', 'SGD', 'RMSprop'])
>     weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
>
>     optimizer = getattr(optim, optimizer_name)(
>         model.parameters(), lr=lr, weight_decay=weight_decay)
>
>     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
>     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
>
>     # 训练（带剪枝）
>     criterion = nn.CrossEntropyLoss()
>     epochs = 30
>
>     for epoch in range(epochs):
>         model.train()
>         for batch_X, batch_y in train_loader:
>             optimizer.zero_grad()
>             output = model(batch_X)
>             loss = criterion(output, batch_y)
>             loss.backward()
>             optimizer.step()
>
>         # 验证
>         model.eval()
>         with torch.no_grad():
>             val_output = model(torch.FloatTensor(X_val))
>             val_pred = val_output.argmax(dim=1)
>             val_acc = (val_pred == torch.LongTensor(y_val)).float().mean().item()
>
>         # 报告并检查剪枝
>         trial.report(val_acc, epoch)
>         if trial.should_prune():
>             raise optuna.TrialPruned()
>
>     return val_acc
>
> # 3. 运行优化
> study = optuna.create_study(
>     direction='maximize',
>     pruner=optuna.pruners.HyperbandPruner(
>         min_resource=5, max_resource=30, reduction_factor=3
>     )
> )
> study.optimize(objective, n_trials=100, timeout=600)
>
> # 4. 结果分析
> print(f"最优验证准确率: {study.best_value:.4f}")
> print(f"\n最优超参数配置:")
> for key, value in study.best_params.items():
>     print(f"  {key}: {value}")
>
> # 5. 使用最优参数重建最终模型
> print(f"\n最优trial编号: {study.best_trial.number}")
> ```


## 六、Optuna 最佳实践


1. **先粗后细：**
   先用少量trial和短训练时间粗略搜索，再在好的区域精细搜索
2. **使用对数尺度：**
   学习率、正则化系数等跨越数量级的参数用 log=True
3. **启用剪枝：**
   深度学习场景必须使用剪枝，推荐 HyperbandPruner
4. **保存Study：**
   使用数据库后端持久化，支持中断恢复和分布式
5. **关注重要参数：**
   根据参数重要性分析，减少不重要参数的搜索范围
6. **条件搜索空间：**
   某些参数只在特定条件下有意义，用 if 语句定义


> **Important:** **常见错误：**
> 不要在 objective 函数中使用全局变量存储最优模型！每个trial是独立的。应该用 callback 机制保存最优模型：
>
> ```
> class SaveBestModelCallback:
>     def __init__(self):
>         self.best_value = None
>         self.best_model = None
>
>     def __call__(self, study, trial):
>         if study.best_trial == trial:
>             self.best_value = trial.value
>             # 在这里保存模型权重
> ```


## 总结


- Optuna的三大核心概念：Study（优化过程）、Trial（一次评估）、Objective（目标函数）
- 搜索空间支持整数、浮点数（含对数尺度）、类别和条件定义
- 剪枝策略可以提前终止不佳的trial，大幅节省计算资源
- 内置可视化功能帮助理解优化过程和参数重要性
- 通过数据库后端支持分布式搜索
- PyTorch/TensorFlow + Optuna 是深度学习超参数优化的标准组合


<!-- Converted from: 02_Optuna与实践.html -->
