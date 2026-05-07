# 3_可微分架构搜索 (DARTS)

## 1. DARTS 核心思想

**DARTS (Differentiable Architecture Search, Liu et al., ICLR 2019)** 将离散的架构搜索问题松弛为**连续优化**，通过梯度下降同时优化网络权重和架构参数。

核心创新：**松弛化搜索空间 + 双层优化**。

## 2. 松弛化搜索空间

### 2.1 从离散到连续

原始问题：每条边选择一个操作（离散）

$$\bar{o}^{(i,j)} = \arg\max_{o \in \mathcal{O}} \text{score}(o)$$

松弛为：每条边是所有操作的加权混合

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

其中 $\alpha^{(i,j)} \in \mathbb{R}^{|\mathcal{O}|}$ 是边 $(i,j)$ 上的**架构参数**（可学习）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """混合操作：所有候选操作的加权组合"""
    def __init__(self, C, stride, candidate_ops):
        super().__init__()
        self.ops = nn.ModuleList([
            op(C, stride) for op in candidate_ops
        ])
    
    def forward(self, x, weights):
        """
        x: 输入特征
        weights: softmax后的架构权重 (num_ops,)
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops))
```

### 2.2 完整 Cell 结构

```python
class DARTSCell(nn.Module):
    """DARTS Cell"""
    def __init__(self, C, num_nodes=4, candidate_ops=None):
        super().__init__()
        self.num_nodes = num_nodes
        
        if candidate_ops is None:
            candidate_ops = [
                nn.ZeroPad2d,  # zero
                lambda C, s: nn.MaxPool2d(3, s, 1),  # max_pool_3x3
                lambda C, s: nn.AvgPool2d(3, s, 1),  # avg_pool_3x3
                lambda C, s: SepConv(C, C, 3, s, 1),  # sep_conv_3x3
                lambda C, s: SepConv(C, C, 5, s, 2),  # sep_conv_5x5
                lambda C, s: DilConv(C, C, 3, s, 2, 2),  # dil_conv_3x3
                lambda C, s: DilConv(C, C, 5, s, 4, 2),  # dil_conv_5x5
                Identity,  # skip_connect
            ]
        
        self.num_ops = len(candidate_ops)
        self.edges = nn.ModuleDict()
        
        for i in range(1, num_nodes):
            for j in range(i + 1):  # 输入节点 + 中间节点
                self.edges[f'{j}_{i}'] = MixedOp(C, 1, candidate_ops)
        
        # 架构参数（不参与权重优化）
        self.alpha = nn.ParameterDict()
        for i in range(1, num_nodes):
            for j in range(i + 1):
                key = f'{j}_{i}'
                self.alpha[key] = nn.Parameter(1e-3 * torch.randn(self.num_ops))
    
    def forward(self, s0, s1):
        """
        s0: 上上层Cell输出
        s1: 上层Cell输出
        """
        states = [s0, s1]
        
        for i in range(1, self.num_nodes):
            # 每个中间节点是所有前置节点的加权混合
            node_inputs = []
            for j in range(i + 1):
                key = f'{j}_{i}'
                weights = F.softmax(self.alpha[key], dim=0)
                edge_out = self.edges[key](states[j], weights)
                node_inputs.append(edge_out)
            
            states.append(sum(node_inputs))
        
        return states[-1]  # 返回最后一个节点的输出
```

## 3. 双层优化 (Bilevel Optimization)

### 3.1 数学形式

DARTS 的优化目标是一个**双层优化问题**：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

- $\alpha$：架构参数（外层优化）
- $w$：网络权重（内层优化）
- $\mathcal{L}_{val}$：验证集损失
- $\mathcal{L}_{train}$：训练集损失

### 3.2 一阶近似

由于精确求解 $w^*(\alpha)$ 需要完整训练，DARTS 使用**一阶近似**：

$$\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{val}(w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha), \alpha)$$

即：**只用一步梯度下降来近似内层优化**。

```python
def darts_train(train_loader, val_loader, model, w_optimizer, alpha_optimizer):
    """DARTS双层优化训练"""
    for epoch in range(num_epochs):
        for (train_input, train_target), (val_input, val_target) in \
                zip(train_loader, val_loader):
            
            # ====== 内层优化：更新权重 w ======
            # 在训练集上计算w的梯度
            w_optimizer.zero_grad()
            train_loss = F.cross_entropy(model(train_input), train_target)
            train_loss.backward()
            
            # 保存当前w的梯度
            w_grads = [p.grad.clone() for p in model.weight_params()]
            
            # 一步近似：w' = w - ξ * ∇_w L_train
            w_prime = [p.data - w_optimizer.param_groups[0]['lr'] * g 
                       for p, g in zip(model.weight_params(), w_grads)]
            
            # ====== 外层优化：更新架构参数 α ======
            alpha_optimizer.zero_grad()
            
            # 用w'计算验证集损失
            # （实际实现中，通常直接复用当前w，省去w'的计算）
            val_loss = F.cross_entropy(model(val_input), val_target)
            val_loss.backward()
            
            # 恢复w的梯度（因为在外层优化中w被视为常数）
            for p, g in zip(model.weight_params(), w_grads):
                p.grad = g
            
            # 更新w
            w_optimizer.step()
            
            # 更新α
            alpha_optimizer.step()
```

### 3.3 二阶近似 (更精确但更慢)

$$\nabla_\alpha \mathcal{L}_{val} \approx \nabla_\alpha \mathcal{L}_{val} - \xi \nabla^2_{\alpha, w} \mathcal{L}_{train} \cdot \nabla_{w'} \mathcal{L}_{val}$$

需要计算Hessian向量积，成本是一阶的2倍。

## 4. 离散化：从连续到最终架构

### 4.1 选择最优操作

训练完成后，每条边保留权重最大的操作：

```python
def derive_architecture(model):
    """从连续架构参数推导离散架构"""
    final_arch = {}
    
    for key, alpha in model.alpha.items():
        # softmax得到权重
        weights = F.softmax(alpha, dim=0)
        # 选择最大权重的操作
        best_op_idx = weights.argmax().item()
        final_arch[key] = best_op_idx
    
    return final_arch
```

### 4.2 保留Top-K边

对于每个节点，保留输入权重最大的前 $K$ 条边（通常 $K=2$）。

## 5. DARTS 的问题与改进

### 5.1 权重共享陷阱

**问题**：DARTS 倾向于选择**跳跃连接 (skip_connect)**，因为它允许梯度直接流过，导致权重训练更快——但不一定最好。

### 5.2 坍塌问题 (Collapse)

DARTS 经常搜索到全部是跳跃连接的架构，验证精度为随机水平。

**原因**：softmax权重在训练后期趋于one-hot，跳过连接的权重被过度放大。

### 5.3 改进方法

| 方法 | 思路 |
|------|------|
| Fair DARTS | 用Sigmoid替代Softmax，消除不公平竞争 |
| DARTS- | 加入Hessian正则化 |
| P-DARTS | 逐步增加搜索深度 |
| PC-DARTS | 部分通道连接，减少内存 |
| SDARTS | 随机扰动增强鲁棒性 |

```python
class FairDARTSCell(DARTSCell):
    """Fair DARTS：用Sigmoid替代Softmax"""
    def forward(self, s0, s1):
        states = [s0, s1]
        
        for i in range(1, self.num_nodes):
            node_inputs = []
            for j in range(i + 1):
                key = f'{j}_{i}'
                # Fair DARTS: Sigmoid而非Softmax
                weights = torch.sigmoid(self.alpha[key])
                edge_out = self.edges[key](states[j], weights)
                node_inputs.append(edge_out)
            
            states.append(sum(node_inputs))
        
        return states[-1]
```

## 6. DARTS 变体对比

| 方法 | 年份 | 改进点 | 计算成本 |
|------|------|--------|----------|
| DARTS (1st) | 2019 | 一阶近似 | 1.5 GPU天 |
| DARTS (2nd) | 2019 | 二阶近似 | 4 GPU天 |
| Fair DARTS | 2020 | Sigmoid替代 | 1.5 GPU天 |
| PC-DARTS | 2020 | 部分通道 | 0.1 GPU天 |
| DARTS- | 2020 | Hessian正则 | 1.5 GPU天 |
| GDAS | 2019 | Gumbel采样 | 0.3 GPU天 |

## 7. 代码实战

```python
import torch
from darts_api import DARTS

# 定义搜索空间
search_space = {
    'num_cells': 8,
    'num_nodes': 4,
    'init_channels': 16,
    'ops': ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 
            'dil_conv_5x5', 'max_pool_3x3', 'avg_pool_3x3', 
            'skip_connect', 'zero'],
}

# 创建模型
model = DARTS(
    C=search_space['init_channels'],
    num_classes=10,
    num_cells=search_space['num_cells'],
    num_nodes=search_space['num_nodes'],
)

# 分离权重参数和架构参数
w_params = model.weight_params()
alpha_params = model.architecture_params()

w_optimizer = torch.optim.SGD(w_params, lr=0.025, momentum=0.9, weight_decay=3e-4)
alpha_optimizer = torch.optim.Adam(alpha_params, lr=3e-4, weight_decay=1e-3)

# 训练
for epoch in range(50):
    darts_train(train_loader, val_loader, model, w_optimizer, alpha_optimizer)

# 推导架构
arch = derive_architecture(model)
print("搜索到的架构:", arch)
```

---

**关键要点**：
1. DARTS 将离散架构搜索松弛为连续优化，用梯度下降同时优化权重和架构参数
2. 双层优化是核心：内层优化权重，外层优化架构
3. 权重共享陷阱是DARTS的主要问题：跳过连接被过度偏好
4. Fair DARTS、PC-DARTS 等变体改善了稳定性和效率
