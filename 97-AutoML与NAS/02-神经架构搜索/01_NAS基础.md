# NAS基础


## 一、NAS 的动机


神经架构搜索（Neural Architecture Search, NAS）的目标是**自动设计神经网络架构**，替代人工试错的过程。


### 1.1 手工设计的局限


- 需要大量领域知识和经验
- 搜索空间巨大：层数、类型、连接方式、核大小等组合爆炸
- 不同任务的最优架构可能截然不同
- 人类偏见可能遗漏某些有效的架构模式


### 1.2 NAS 的三大核心组件


| 组件 | 作用 | 代表方法 |
| --- | --- | --- |
| **搜索空间** | 定义可能的架构集合 | 链式、多分支、cell-based |
| **搜索策略** | 如何探索搜索空间 | RL、进化算法、梯度 |
| **性能评估** | 如何评估架构好坏 | 训练后评估、权重共享 |


## 二、搜索空间


### 2.1 链式结构搜索空间


最简单的搜索空间：网络由一系列层依次连接，每层从候选操作中选择一种。


- 候选操作：Conv3x3, Conv5x5, MaxPool3x3, AvgPool3x3, Skip, Identity
- 搜索维度：层数 × 每层候选操作数
- 局限：无法表达跳跃连接和多分支结构


### 2.2 多分支结构搜索空间


每层可以有多个输入，通过加权合并。例如 NASNet 的搜索空间。


### 2.3 Cell-based 搜索空间


这是目前最主流的方法：只搜索**小的构建块（Cell）**的结构，然后将Cell堆叠成完整网络。


$$
一个Cell包含 N 个节点，每个节点 = 两个输入的组合操作
                搜索空间大小 ≈ (候选操作数 × 候选输入数)N
                NASNet: 13种操作 × C(5,2)输入 × 5个节点 ≈ 5.6 × 1014 种架构
$$


| 搜索空间类型 | 搜索范围 | 灵活性 | 搜索难度 |
| --- | --- | --- | --- |
| 链式 | 每层的操作类型 | 低 | 简单 |
| 多分支 | 连接+操作 | 中 | 中等 |
| Cell-based | Cell内部结构 | 高 | 较难 |
| 层级Cell | Normal Cell + Reduction Cell | 高 | 较难 |


## 三、搜索策略


### 3.1 强化学习（RL）— NASNet


Zoph & Le (2017) 的开创性工作：用RNN控制器生成架构描述，训练后的验证准确率作为奖励信号。


$$
控制器 RNN 生成架构 α
                训练子网络得到准确率 R(α)
                用 REINFORCE 算法更新控制器参数 θ：
                ∇θ J(θ) ≈ (R(α) - b) · ∇θ log P(α; θ)
$$


- NASNet 在 CIFAR-10 上达到当时的 SOTA
- **代价惊人：**
   使用 800 块 GPU，训练了 28 天


### 3.2 进化算法


Real et al. (2019) 用进化算法搜索架构：


1. 初始化种群（随机架构）
2. 评估适应度（训练后准确率）
3. 选择、变异（添加/删除层、改变操作）
4. 迭代直到计算预算耗尽


AmoebaNet 用进化方法在 ImageNet 上超过了 NASNet，且计算量更少。


### 3.3 梯度方法 — DARTS


DARTS（Differentiable Architecture Search）是最重要的突破之一，将离散搜索空间松弛为连续，用梯度下降优化。


## 四、DARTS：可微分架构搜索


### 4.1 核心思想


原始NAS的搜索空间是离散的（选择某个操作），DARTS将其松弛为连续：每个边上的操作是所有候选操作的**加权混合**。


$$
松弛操作：
                õ(i,j)(x) = ∑o∈O softmax(αo(i,j)) · o(x)
                其中 αo(i,j) 是可学习的架构参数
$$


### 4.2 双层优化


$$
外层优化（架构参数 α）：
                minα Lval(w*(α), α)
                内层优化（网络权重 w）：
                w*(α) = argminw Ltrain(w, α)
$$


### 4.3 DARTS 的优势与问题


| 优势 | 问题 |
| --- | --- |
| 搜索速度快（单GPU 1天） | 性能不稳定，多次运行差异大 |
| 内存效率高（权重共享） | 倾向于skip connection（跳跃连接过多） |
| 端到端可微分 | 搜索-评估gap（proxy差距） |


> **Important:** **DARTS的后续改进：**
> PC-DARTS（部分通道连接）、Fair DARTS（公平性约束）、DARTS-（稳定性改进）、FairDARTS等。这些改进主要解决DARTS的不稳定性和skip connection倾向问题。


## 五、性能评估策略


NAS最大的计算瓶颈是评估每个候选架构的性能。传统方法需要从头训练每个架构，成本极高。


| 评估方法 | 原理 | 加速比 | 精度损失 |
| --- | --- | --- | --- |
| 完整训练 | 从头训练到收敛 | 1x | 无 |
| 早停（Early Stopping） | 训练少量epoch | 10-50x | 有，排名可能不准 |
| 权重共享（One-Shot） | 所有架构共享一个超网 | 1000x+ | 较大 |
| 代理模型 | 用学习曲线预测最终性能 | 10-100x | 中等 |
| 零成本代理 | 不训练，直接用初始化指标 | 极大 | 粗略估计 |


> **Note:** **权重共享的One-Shot方法：**
> 训练一个包含所有可能操作的超网络（SuperNet），评估子架构时直接继承超网络的权重，无需单独训练。DARTS就是这种方法的代表。虽然加速显著，但存在排名不一致的问题。


## 六、NAS 的计算成本演进


| 方法 | 年份 | GPU天数 | CIFAR-10错误率 |
| --- | --- | --- | --- |
| NASNet | 2017 | 22400 | 2.65% |
| AmoebaNet | 2019 | 3150 | 2.13% |
| DARTS | 2019 | 1.5 | 2.76% |
| ProxylessNAS | 2019 | 8.3 | 2.08% |
| EfficientNAS | 2020 | 0.4 | 2.0% |


> **Note:** **趋势：**
> NAS的计算成本从数千GPU天降到不到1天，但搜索出的架构性能并未显著下降。这得益于更高效的搜索策略（DARTS）和更好的评估方法（权重共享、早停）。


## 七、Python 实战：DARTS 搜索示例


> **Example:** ### 示例：使用 darts 库进行架构搜索
>
>
> ```
> # 安装: pip install dartsnas
>
> import torch
> import torch.nn as nn
> from darts import api
> from darts.search_spaces import CNNSearchSpace
>
> # 1. 定义搜索空间
> class SimpleNASSearchSpace(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.ops = nn.ModuleList([
>             nn.Conv2d(16, 16, 3, padding=1),
>             nn.Conv2d(16, 16, 5, padding=2),
>             nn.MaxPool2d(3, stride=1, padding=1),
>             nn.Identity(),
>         ])
>         self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
>
>     def forward(self, x):
>         weights = torch.softmax(self.alpha, dim=0)
>         return sum(w * op(x) for w, op in zip(weights, self.ops))
>
> # 2. 简化的DARTS搜索循环
> def darts_search(model, train_loader, val_loader, epochs=50):
>     optimizer_w = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
>     optimizer_a = torch.optim.Adam([model.alpha], lr=3e-4, weight_decay=1e-3)
>     criterion = nn.CrossEntropyLoss()
>
>     for epoch in range(epochs):
>         # 交替优化权重和架构参数
>         model.train()
>         for (train_x, train_y), (val_x, val_y) in zip(train_loader, val_loader):
>             # 更新架构参数（在验证集上）
>             optimizer_a.zero_grad()
>             val_pred = model(val_x)
>             loss_a = criterion(val_pred, val_y)
>             loss_a.backward()
>             optimizer_a.step()
>
>             # 更新网络权重（在训练集上）
>             optimizer_w.zero_grad()
>             train_pred = model(train_x)
>             loss_w = criterion(train_pred, train_y)
>             loss_w.backward()
>             optimizer_w.step()
>
>         if (epoch + 1) % 10 == 0:
>             print(f"Epoch {epoch+1}: arch_params={torch.softmax(model.alpha, 0).detach()}")
>
>     # 3. 导出最终架构
>     best_op = torch.argmax(model.alpha).item()
>     print(f"最优操作索引: {best_op}")
>     return best_op
> ```


## 总结


- NAS的三大组件：搜索空间、搜索策略、性能评估
- Cell-based搜索空间是当前主流，搜索小模块再堆叠成网络
- 搜索策略从RL(NASNet)演进到梯度方法(DARTS)，计算成本降低万倍
- DARTS将离散搜索松弛为连续优化，但存在不稳定和skip connection问题
- 权重共享(One-Shot)大幅降低评估成本，但有排名不一致的问题
- NAS的计算成本从22400 GPU天降至不到1天


<!-- Converted from: 01_NAS基础.html -->
