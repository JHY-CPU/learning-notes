# 2_基于强化学习的NAS

## 1. RL-NAS 基本思想

**NASNet (Zoph & Le, ICLR 2018)** 首次将强化学习应用于NAS，用**控制器RNN**作为智能体生成网络架构，将验证精度作为奖励信号。

### 1.1 将NAS建模为RL问题

| RL概念 | NAS对应 |
|--------|---------|
| 环境 | 训练/验证数据集 |
| 状态 | 当前已生成的部分架构 |
| 动作 | 选择下一个操作（卷积类型、核大小等） |
| 奖励 | 验证集精度 |
| 策略 | 控制器RNN |

### 1.2 控制器架构

```
Controller RNN (LSTM)
    ↓
预测 Cell 结构:
    Step 1: 选择隐藏状态B_1 → 选操作op_1 → 选输入connection_1
    Step 2: 选择隐藏状态B_2 → 选操作op_2 → 选输入connection_2
    ...
    Step N: 选择隐藏状态B_N → 选操作op_N → 选输入connection_N
    ↓
组装 Cell → 堆叠成完整网络 → 训练 → 验证精度 → 奖励
```

## 2. NASNet

### 2.1 Cell 搜索

NASNet搜索两种Cell：

1. **Normal Cell**：不改变空间分辨率
2. **Reduction Cell**：将空间分辨率减半

每个Cell有 $B=5$ 个Block（块），每个Block：

```
1. 选择两个隐藏状态 (候选: 前面所有Block的输出 + Cell输入)
2. 对每个隐藏状态选择一个操作
3. 选择合并方式 (加法 or 拼接)
```

### 2.2 控制器实现

```python
class NASNetController(nn.Module):
    """NASNet控制器RNN"""
    def __init__(self, num_blocks=5, hidden_size=100, 
                 num_ops=13, num_candidates=7):
        super().__init__()
        self.num_blocks = num_blocks
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        # 输出层：每步预测
        self.op_embedding = nn.Embedding(num_ops, hidden_size)
        self.fc_op = nn.Linear(hidden_size, num_ops)
        self.fc_input1 = nn.Linear(hidden_size, num_candidates)
        self.fc_input2 = nn.Linear(hidden_size, num_candidates)
        self.fc_combine = nn.Linear(hidden_size, 2)  # add or concat
    
    def forward(self, num_cells=2):
        """生成架构描述"""
        architectures = []
        
        for cell_type in ['normal', 'reduction']:
            hx = torch.zeros(1, self.lstm.hidden_size)
            cx = torch.zeros(1, self.lstm.hidden_size)
            
            blocks = []
            for b in range(self.num_blocks):
                # 输入1
                input1_logits = self.fc_input1(hx)
                input1 = F.gumbel_softmax(input1_logits, tau=1.0, hard=True).argmax()
                
                # 操作1
                op1_logits = self.fc_op(hx)
                op1 = F.gumbel_softmax(op1_logits, tau=1.0, hard=True).argmax()
                
                # 输入2
                hx, cx = self.lstm(self.op_embedding(op1), (hx, cx))
                input2_logits = self.fc_input2(hx)
                input2 = F.gumbel_softmax(input2_logits, tau=1.0, hard=True).argmax()
                
                # 操作2
                op2_logits = self.fc_op(hx)
                op2 = F.gumbel_softmax(op2_logits, tau=1.0, hard=True).argmax()
                
                # 合并方式
                hx, cx = self.lstm(self.op_embedding(op2), (hx, cx))
                combine_logits = self.fc_combine(hx)
                combine = F.gumbel_softmax(combine_logits, tau=1.0, hard=True).argmax()
                
                blocks.append({
                    'input1': input1.item(), 'op1': op1.item(),
                    'input2': input2.item(), 'op2': op2.item(),
                    'combine': combine.item()  # 0=add, 1=concat
                })
                
                hx, cx = self.lstm(self.op_embedding(op2), (hx, cx))
            
            architectures.append({'type': cell_type, 'blocks': blocks})
        
        return architectures
```

### 2.3 REINFORCE 算法训练控制器

```python
def train_controller(controller, dataset, num_iterations=800):
    """用REINFORCE训练控制器"""
    baseline = 0  # 滑动平均基线
    
    for iteration in range(num_iterations):
        # 1. 采样架构
        architecture = controller()
        
        # 2. 构建并训练子网络
        child_model = build_network(architecture)
        accuracy = train_and_evaluate(child_model, dataset, epochs=20)
        
        # 3. 更新基线（滑动平均）
        baseline = 0.9 * baseline + 0.1 * accuracy
        
        # 4. REINFORCE梯度
        # ∇θ J(θ) = E[(R - b) ∇θ log π(a|s)]
        reward = accuracy - baseline
        
        # 计算控制器梯度（需要记录每个动作的log概率）
        log_probs = controller.get_log_probs(architecture)
        policy_loss = -reward * log_probs
        
        # 5. 更新控制器
        optimizer.zero_grad()
        policy_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=0.5)
        optimizer.step()
        
        # 6. 记录最优
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_architecture = architecture
```

## 3. ENAS (Efficient NAS)

### 3.1 核心创新

**ENAS (Pham et al., ICLR 2018)** 解决了NASNet的计算瓶颈：**所有架构共享权重**。

```
NASNet: 每个候选架构独立训练 → 2000 GPU天
ENAS:  所有架构共享一个超网络权重 → 0.5 GPU天
```

### 3.2 权重共享超网络

```python
class ENASSuperNet(nn.Module):
    """ENAS超网络：所有架构共享权重"""
    def __init__(self, num_nodes=12, candidate_ops=8):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 所有候选操作的权重（共享）
        self.shared_ops = nn.ModuleList([
            SepConv3x3(), SepConv5x5(),
            DilConv3x3(), DilConv5x5(),
            MaxPool3x3(), AvgPool3x3(),
            Identity(), ZeroOp(),
        ])
    
    def forward(self, x, architecture):
        """
        architecture: 控制器给出的架构决策
        """
        # 存储所有节点的输出
        h = [x]
        
        for node_idx in range(1, self.num_nodes):
            # 获取该节点的架构决策
            prev_node = architecture[f'node_{node_idx}_input']
            op_idx = architecture[f'node_{node_idx}_op']
            
            # 使用共享权重执行操作
            h_new = self.shared_ops[op_idx](h[prev_node])
            h.append(h_new)
        
        return h[-1]
```

### 3.3 联合训练

```python
def train_enas(controller, supernet, dataset, num_epochs=300):
    """ENAS联合训练：交替更新控制器和超网络"""
    
    for epoch in range(num_epochs):
        # ====== 阶段1：训练超网络权重 ======
        # 冻结控制器，采样架构训练超网络
        for batch in dataloader:
            # 控制器采样架构
            with torch.no_grad():
                architecture = controller.sample()
            
            # 前向传播（使用采样的架构）
            pred = supernet(batch['input'], architecture)
            loss = F.cross_entropy(pred, batch['label'])
            
            # 更新超网络权重
            supernet_optimizer.zero_grad()
            loss.backward()
            supernet_optimizer.step()
        
        # ====== 阶段2：训练控制器 ======
        # 冻结超网络，更新控制器
        for _ in range(2000):
            # 采样架构
            architecture, log_prob = controller.sample_with_log_prob()
            
            # 评估（用超网络权重）
            with torch.no_grad():
                acc = evaluate_with_shared_weights(supernet, architecture, val_set)
            
            # REINFORCE更新
            reward = acc - baseline
            policy_loss = -reward * log_prob
            
            controller_optimizer.zero_grad()
            policy_loss.backward()
            controller_optimizer.step()
```

## 4. 搜索结果

### 4.1 NASNet 搜索到的 Cell

NASNet-A Cell 结构（在CIFAR-10上搜索，在ImageNet上迁移）：

```
Normal Cell:
  Block 1: B_1 = SepConv5x5(h_{t-1})  
  Block 2: B_2 = SepConv3x3(h_{t-1})
  Block 3: B_3 = AvgPool3x3(B_1)
  Block 4: B_4 = SepConv3x3(h_{t-1})
  Block 5: B_5 = SepConv3x3(B_2)
```

### 4.2 计算成本对比

| 方法 | GPU天数 | ImageNet Top-1 |
|------|---------|----------------|
| NASNet-A | 2000 | 74.0% |
| AmoebaNet-A | 3150 | 74.3% |
| ENAS | 0.5 | 74.3% |
| DARTS | 1.5 | 73.3% |
| PNAS | 225 | 74.2% |

## 5. 优缺点

### 5.1 优势

1. **端到端自动设计**：无需人工设计架构
2. **可发现新结构**：NASNet发现了SE模块等新设计
3. **可迁移**：搜索的Cell可迁移到不同任务/数据集

### 5.2 局限

1. **计算成本**：原始NASNet需要2000 GPU天
2. **搜索空间依赖**：结果依赖于预定义的搜索空间
3. **离散优化**：架构空间是离散的，梯度方法不直接适用
4. **搜索偏见**：RL控制器可能偏好简单架构

---

**关键要点**：
1. NASNet首次将NAS建模为RL问题，用控制器RNN生成架构
2. REINFORCE算法是最常用的控制器训练方法
3. ENAS通过权重共享将搜索成本从2000 GPU天降到0.5天
4. 权重共享是效率的关键，但也引入了评估偏差
