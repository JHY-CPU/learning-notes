# 10_数据异构性与 Non-IID 问题

## 1. Non-IID 问题概述

在联邦学习中，不同客户端的数据通常**不是独立同分布 (Non-IID)** 的。这是联邦学习区别于传统分布式学习的最关键特征，也是联邦学习面临的最大挑战之一。

### 1.1 IID vs Non-IID

```
IID (独立同分布):
  客户端 1: [猫, 狗, 鸟, 猫, 狗, ...]  (均匀分布)
  客户端 2: [猫, 狗, 鸟, 猫, 狗, ...]  (均匀分布)
  客户端 3: [猫, 狗, 鸟, 猫, 狗, ...]  (均匀分布)

Non-IID (非独立同分布):
  客户端 1: [猫, 猫, 猫, 猫, ...]       (标签偏斜)
  客户端 2: [狗, 狗, 狗, 狗, ...]       (标签偏斜)
  客户端 3: [鸟, 鸟, 鸟, 鸟, ...]       (标签偏斜)
```

### 1.2 Non-IID 类型分类

| 类型 | 描述 | 示例 |
|------|------|------|
| 标签分布偏斜 | 每个客户端只有部分类别 | 医院只有特定疾病的数据 |
| 特征分布偏斜 | 相同标签但特征分布不同 | 不同地区的手写风格 |
| 数量偏斜 | 客户端数据量差异大 | 活跃用户 vs 非活跃用户 |
| 概念偏斜 | 同一样本在不同客户端有不同标签 | 不同标注者的差异 |
| 时间偏斜 | 数据随时间变化 | 季节性数据 |

## 2. Non-IID 的影响

### 2.1 性能下降

```python
def simulate_non_iid_effect(model, iid_data, non_iid_data, num_rounds=100):
    """
    模拟 IID vs Non-IID 下 FedAvg 的性能差异
    """
    results = {'iid': [], 'non_iid': []}

    for scenario, data in [('iid', iid_data), ('non_iid', non_iid_data)]:
        fedavg = FedAvgServer(copy.deepcopy(model))
        accs = fedavg.run(data, num_rounds)
        results[scenario] = accs

    # 典型结果:
    # IID 最终准确率: 92%
    # Non-IID 最终准确率: 68%  ← 大幅下降
    return results
```

### 2.2 原因分析

```
Non-IID 导致性能下降的原因:

1. 梯度冲突
   不同客户端的梯度方向差异大
   全局平均后可能互相抵消

2. 客户端漂移 (Client Drift)
   每个客户端向自己的最优方向训练
   全局模型被拉向不同方向

3. 收敛困难
   全局最优 ≠ 各本地最优的平均
   需要更多通信轮数
```

## 3. FedProx 解决方案

### 3.1 算法原理

Li 等人 (2020) 提出 FedProx，在本地训练时添加**近端项 (Proximal Term)**，限制本地模型与全局模型的偏离：

$$\min_w \mathcal{L}_k(w) + \frac{\mu}{2} \|w - w_t\|^2$$

其中 $\mu$ 是近端正则化强度。

### 3.2 实现

```python
class FedProxClient:
    """
    FedProx 客户端: 添加近端正则化
    """
    def __init__(self, client_id, local_data, local_epochs=5,
                 lr=0.01, mu=0.01):
        self.client_id = client_id
        self.local_data = local_data
        self.local_epochs = local_epochs
        self.lr = lr
        self.mu = mu  # 近端正则化强度

    def local_train(self, model, global_model):
        """
        FedProx 本地训练
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        # 保存全局模型参数 (用于近端正则化)
        global_params = {
            name: param.clone().detach()
            for name, param in global_model.named_parameters()
        }

        for epoch in range(self.local_epochs):
            for x, y in self.local_data:
                optimizer.zero_grad()

                # 任务损失
                output = model(x)
                task_loss = nn.functional.cross_entropy(output, y)

                # 近端正则化损失
                prox_loss = 0
                for name, param in model.named_parameters():
                    prox_loss += ((param - global_params[name]) ** 2).sum()
                prox_loss = (self.mu / 2) * prox_loss

                # 总损失
                total_loss = task_loss + prox_loss
                total_loss.backward()
                optimizer.step()

        return model
```

### 3.3 FedProx vs FedAvg 对比

| 特性 | FedAvg | FedProx |
|------|--------|---------|
| 本地目标 | $\min \mathcal{L}_k(w)$ | $\min \mathcal{L}_k(w) + \frac{\mu}{2}\|w-w_t\|^2$ |
| 客户端漂移 | 严重 | 受控 |
| Non-IID 性能 | 差 | **更好** |
| 异构设备 | 需要统一 epoch | 支持部分参与 |
| 额外超参数 | 无 | $\mu$ |

## 4. 其他 Non-IID 解决方案

### 4.1 数据共享策略

```python
def create_shared_dataset(num_classes=10, samples_per_class=10):
    """
    创建服务器端共享的小数据集
    用于辅助全局训练
    """
    shared_data = []
    for c in range(num_classes):
        # 生成或收集少量公共数据
        samples = generate_class_samples(c, samples_per_class)
        shared_data.extend(samples)
    return shared_data
```

### 4.2 局部正则化

```python
class ScaffoldClient:
    """
    SCAFFOLD 客户端: 使用控制变量校正客户端漂移
    """
    def local_train(self, model, global_model, c_local, c_global):
        """
        使用控制变量校正梯度
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        for epoch in range(self.local_epochs):
            for x, y in self.local_data:
                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()

                # 校正梯度: g_corrected = g_local - c_local + c_global
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad += (c_global[name] - c_local[name])

                optimizer.step()

        return model
```

### 4.3 类不平衡处理

```python
def handle_class_imbalance(client_data, num_classes):
    """
    处理客户端的类别不平衡
    """
    # 统计类别分布
    class_counts = torch.zeros(num_classes)
    for _, y in client_data:
        for label in y:
            class_counts[label] += 1

    # 计算类别权重
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-8)

    return weights
```

## 5. Non-IID 数据模拟

```python
def create_non_iid_data(dataset, num_clients, non_iid_type='label_skew',
                         alpha=0.5):
    """
    创建 Non-IID 数据分布

    Args:
        dataset: 完整数据集
        num_clients: 客户端数量
        non_iid_type: Non-IID 类型
        alpha: Dirichlet 分布参数 (越小越不均匀)
    """
    num_classes = len(set(dataset.targets))
    client_data = [[] for _ in range(num_clients)]

    if non_iid_type == 'label_skew':
        # Dirichlet 分布模拟标签偏斜
        for c in range(num_classes):
            # 从 Dirichlet 分布采样分配比例
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * sum(dataset.targets == c)).astype(int)

            # 分配样本
            class_indices = [i for i, y in enumerate(dataset.targets) if y == c]
            idx = 0
            for client_id, count in enumerate(proportions):
                client_data[client_id].extend(
                    class_indices[idx:idx+count]
                )
                idx += count

    elif non_iid_type == 'quantity_skew':
        # 数据量不均匀
        total = len(dataset)
        proportions = np.random.dirichlet([alpha] * num_clients)
        for client_id in range(num_clients):
            count = int(proportions[client_id] * total)
            client_data[client_id] = list(range(
                sum(int(proportions[i] * total) for i in range(client_id)),
                sum(int(proportions[i] * total) for i in range(client_id+1))
            ))

    return client_data
```

## 6. 总结

| 要点 | 说明 |
|------|------|
| Non-IID 类型 | 标签偏斜、特征偏斜、数量偏斜 |
| 影响 | 性能下降、收敛变慢、客户端漂移 |
| FedProx | 近端正则化限制本地偏离 |
| 其他方案 | SCAFFOLD、数据共享、类权重 |

---

**参考文献：**

1. Zhao, Y. et al. (2018). *Federated learning with non-IID data*. arXiv.
2. Li, T. et al. (2020). *Federated optimization in heterogeneous networks (FedProx)*. MLSys.
3. Karimireddy, S. et al. (2020). *SCAFFOLD: Stochastic controlled averaging for federated learning*. ICML.
