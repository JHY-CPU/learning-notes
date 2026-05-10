# 多目标优化与MMOE


## 1. 推荐系统中的多目标


### 1.1 为什么需要多目标


工业推荐系统需要同时优化多个业务指标，单一目标无法全面衡量推荐质量。例如电商推荐不仅关心用户是否点击（CTR），还关心是否购买（CVR）、用户停留时长、收藏、分享等。


### 1.2 常见多目标场景


| 场景 | 目标1 | 目标2 | 目标3 | 目标4 |
| --- | --- | --- | --- | --- |
| 电商 | CTR（点击率） | CVR（转化率） | 停留时长 | 收藏/加购 |
| 短视频 | 完播率 | 点赞率 | 评论率 | 关注率 |
| 新闻 | CTR | 阅读时长 | 分享率 | 负反馈率 |
| 广告 | CTR | CVR | GMV | 广告收入 |


### 1.3 多目标融合的方式


最终的推荐分数通常由多个目标的预测值加权融合：


$$
score = w1 · pCTR + w2 · pCVR + w3 · pStayTime + ...
                其中权重 wi 可以通过人工调参、在线学习或贝叶斯优化来确定
$$


> **Note:** **ESMM模型（阿里）：**
> 针对CTR和CVR联合建模，解决CVR的样本选择偏差问题。通过建模CTCVR = P(click) × P(conversion|click)，利用全空间的曝光数据训练CVR，而非只在点击样本上训练。


## 2. 多任务学习（Multi-Task Learning）


### 2.1 核心动机


- **参数共享**
   ：多个任务共享底层特征表示，减少总参数量
- **隐式数据增强**
   ：不同任务的噪声模式不同，多任务学习相当于正则化
- **信息迁移**
   ：辅助任务可以为目标任务提供有用信息
- **工程效率**
   ：一个模型同时预测多个目标，减少维护成本


### 2.2 Hard Parameter Sharing


所有任务共享底层的隐层参数，只在最后一层分叉出不同的任务塔：


```
输入 → [共享隐层1] → [共享隐层2] → [共享隐层3]
                                    ├→ [任务1塔] → 输出1
                                    ├→ [任务2塔] → 输出2
                                    └→ [任务3塔] → 输出3
```


- **优点**
   ：参数少，过拟合风险低
- **缺点**
   ：任务冲突时效果差，负迁移


### 2.3 Soft Parameter Sharing


每个任务有自己的参数，但通过正则化约束不同任务的参数尽可能接近：


- 各任务有独立的网络
- 通过L2正则化或注意力机制进行参数约束
- 灵活性更高，但参数量大


## 3. MMOE（Multi-gate Mixture-of-Experts）


### 3.1 MMOE结构


MMOE（Google, 2018）是工业界最广泛使用的多任务学习框架。其核心思想是用多个Expert（专家网络）和每个任务一个Gate（门控网络）来实现任务间参数的自适应共享。


```
输入 x → [Expert 1] ─┐
     ├→ [Expert 2] ──┤
     ├→ [Expert 3] ──┼→ [Gate_k] → 加权组合 → [Task k Tower] → 输出_k
     └→ [Expert 4] ──┘    ↑
                      每个任务独立的Gate
```


### 3.2 数学公式


$$
对于第k个任务：
                gk(x) = softmax(Wgk · x)  ← Gate网络输出
                fk(x) = Σi=1n gki(x) · Ei(x)  ← 加权Expert输出
                yk = Towerk(fk(x))  ← 任务塔输出
$$


### 3.3 为什么MMOE有效


- **自适应共享**
   ：Gate网络学习每个任务与各Expert的关系，自动决定如何分配Expert给不同任务
- **缓解负迁移**
   ：相似任务可以共享Expert，冲突任务使用不同Expert
- **参数效率**
   ：Expert数量通常远少于任务数，仍能有效共享


> **Example:** **直觉理解：**
> 假设有4个Expert和3个任务。Expert 1可能学到的是"用户活跃度"相关的特征，Expert 2学到"内容质量"特征，Expert 3学到"时间偏好"特征，Expert 4学到"社交关系"特征。Gate网络发现CTR任务主要需要Expert 1和Expert 2，CVR主要需要Expert 2和Expert 3，停留时长主要需要Expert 1和Expert 4。


## 4. PLE（Progressive Layered Extraction）


### 4.1 PLE的动机


MMOE存在一个明显的问题：当任务之间的相关性较弱时，一个任务的Gate可能被其他任务"带偏"（跷跷板现象）。PLE通过引入**任务专属Expert**和**共享Expert**的显式分离来解决这个问题。


### 4.2 PLE结构


```
输入 x → [Expert 1 专属] ─┐
     ├→ [Expert 2 专属] ──┤
     ├→ [Expert 3 专属] ──┼→ [CGC门控] → 逐层提取 → Task Tower → 输出
     ├→ [Expert S1 共享] ─┤
     └→ [Expert S2 共享] ─┘
```


### 4.3 CGC（Customized Gate Control）


每个任务的Gate只对"自己的专属Expert + 共享Expert"进行加权，不涉及其他任务的Expert：


$$
gk(x) = softmax(Wgk · x)
                fk(x) = Σ gk,i · Experti(x)  (i ∈ 专属Expertk ∪ 共享Expert)
$$


### 4.4 Progressive Layered Extraction


PLE将CGC单元堆叠多层，逐层提取更高级的特征交互：


- 底层：共享信息较多，Expert差异较小
- 高层：通过门控机制逐步分离任务相关和无关信息
- 最终：每个任务得到最适合自己的特征表示


### 4.5 PLE vs MMOE


| 维度 | MMOE | PLE |
| --- | --- | --- |
| Expert划分 | 全共享 | 专属+共享 |
| 门控范围 | 所有Expert | 仅自己的Expert+共享 |
| 弱相关任务 | 可能负迁移 | 通过专属Expert避免 |
| 参数量 | 较少 | 稍多 |
| 效果 | 任务相关时好 | 通用更强 |


## 5. 任务间的相关性与冲突


### 5.1 正相关 vs 负相关


| 类型 | 示例 | 影响 |
| --- | --- | --- |
| 正相关 | CTR和停留时长 | 多任务学习有增益 |
| 弱相关 | CTR和分享率 | 多任务学习有微弱增益 |
| 负相关 | CTR和内容质量 | 可能产生负迁移 |


### 5.2 负迁移（Negative Transfer）


当任务间差异较大时，强制共享参数反而会损害各任务的表现。解决方案：


- 使用PLE的专属Expert机制
- GradNorm：动态调整各任务的梯度权重
- PCGrad：投影冲突梯度，消除负方向
- 根据任务相关性动态调整共享程度


### 5.3 帕累托最优（Pareto Optimality）


在多目标优化中，帕累托最优是指不存在一个解能同时改进所有目标。帕累托前沿（Pareto Front）上所有解都是最优折衷方案。


$$
解A帕累托支配解B ↔ 所有目标上A ≥ B，且至少一个目标A > B
                帕累托最优解 ↔ 不被任何其他解帕累托支配
$$


寻找帕累托最优解的方法：多目标梯度下降（MGDA）、帕累托MTL等。


## 6. PyTorch代码：MMOE实现


```
import torch
import torch.nn as nn

class Expert(nn.Module):
    """单个Expert网络"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    """任务门控网络"""
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.gate(x)  # (batch, num_experts)


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts"""

    def __init__(self, input_dim, num_experts=4, expert_dim=64,
                 num_tasks=2, task_tower_dims=(32,)):
        super(MMOE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Expert网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim)
            for _ in range(num_experts)
        ])

        # 每个任务一个Gate
        self.gates = nn.ModuleList([
            Gate(input_dim, num_experts)
            for _ in range(num_tasks)
        ])

        # 每个任务的Tower网络
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            layers = []
            in_dim = expert_dim
            for h_dim in task_tower_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.towers.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        x: (batch, input_dim)
        returns: list of (batch,) 每个任务的预测
        """
        # 计算所有Expert输出
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )  # (batch, num_experts, expert_dim)

        outputs = []
        for k in range(self.num_tasks):
            # Gate权重
            gate_w = self.gates[k](x)  # (batch, num_experts)

            # 加权Expert组合
            weighted = torch.bmm(
                gate_w.unsqueeze(1),     # (batch, 1, num_experts)
                expert_outs              # (batch, num_experts, expert_dim)
            ).squeeze(1)                 # (batch, expert_dim)

            # 任务Tower
            task_out = self.towers[k](weighted).squeeze(-1)
            outputs.append(task_out)

        return outputs


# 使用示例
if __name__ == "__main__":
    # 4个Expert，2个任务（CTR和CVR）
    model = MMOE(
        input_dim=128,
        num_experts=4,
        expert_dim=64,
        num_tasks=2,
        task_tower_dims=(32,)
    )

    x = torch.randn(32, 128)  # batch=32, 特征维度=128
    ctr_pred, cvr_pred = model(x)
    print(f"CTR预测: {ctr_pred.shape}")  # (32,)
    print(f"CVR预测: {cvr_pred.shape}")  # (32,)

    # 多任务损失
    ctr_loss = nn.BCEWithLogitsLoss()(ctr_pred, torch.randint(0, 2, (32,)).float())
    cvr_loss = nn.BCEWithLogitsLoss()(cvr_pred, torch.randint(0, 2, (32,)).float())
    total_loss = ctr_loss + cvr_loss
    print(f"总损失: {total_loss.item():.4f}")
```


## 7. 多目标优化最佳实践


### 7.1 损失加权策略


| 策略 | 描述 | 优缺点 |
| --- | --- | --- |
| 固定权重 | 人工设定各任务loss权重 | 简单但需要大量调参 |
| Uncertainty Weighting | 基于同方差不确定性自动学习权重 | 自动调节，但可能不稳定 |
| GradNorm | 根据梯度范数动态调整权重 | 平衡梯度，但计算开销大 |
| DWA | 动态权重平均，基于loss变化率 | 简单有效 |
| Loss-Balanced | 按各任务loss量级归一化 | 防止大loss主导 |


> **Important:** **关键建议：**
>
> - 先训练单任务baseline，了解各任务的收敛情况和难度
> - 观察任务间的相关性，正相关任务多任务学习增益更大
> - Expert数量通常设为任务数的1~2倍
> - 线上融合时需要根据业务目标调整各目标的权重
> - 定期监控各目标的离线指标，避免跷跷板现象


<!-- Converted from: 03_多目标优化与MMOE.html -->
