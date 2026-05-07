# 8_SASRec：自注意力序列推荐

## 1. 概述

SASRec (Self-Attentive Sequential Recommendation, ICDM 2018) 使用单向自注意力建模用户行为序列，预测下一个交互物品。它是 Transformer 在序列推荐中的开创性工作。

**核心思想：** 每个物品在序列中的表示应由其前序物品通过注意力机制动态聚合得到。

## 2. 模型架构

```
输入序列: [item₁, item₂, item₃, item₄, item₅]
              ↓
         物品嵌入 + 位置嵌入
              ↓
    [因果自注意力层] × N
              ↓
         各时间步表示
              ↓
    预测下一个物品 (最后一个位置)
```

```python
import torch
import torch.nn as nn
import numpy as np

class SASRec(nn.Module):
    """SASRec: 自注意力序列推荐"""
    def __init__(self, n_items, max_len=50, embed_dim=64,
                 n_heads=1, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.embed_dim = embed_dim

        # 物品嵌入（+1 留给 padding）
        self.item_embedding = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer 层（使用因果掩码）
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN 更稳定
            )
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq):
        """
        item_seq: (B, L) 物品 ID 序列，0 为 padding
        """
        B, L = item_seq.shape

        # 物品嵌入
        item_emb = self.item_embedding(item_seq)  # (B, L, D)

        # 位置嵌入
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # 合并
        x = self.layer_norm(self.dropout(item_emb + pos_emb))

        # 因果掩码（下三角矩阵，True 表示不参与注意力）
        causal_mask = torch.triu(torch.ones(L, L, device=item_seq.device),
                                  diagonal=1).bool()

        # padding 掩码
        padding_mask = (item_seq == 0)

        # 多层自注意力
        for layer in self.attention_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        return x  # (B, L, D)

    def predict(self, item_seq):
        """预测下一个物品的概率分布"""
        encoded = self.forward(item_seq)  # (B, L, D)

        # 取最后一个有效位置的输出
        last_positions = (item_seq != 0).sum(dim=1) - 1  # (B,)
        last_hidden = encoded[torch.arange(item_seq.size(0)), last_positions]

        # 与所有物品嵌入计算相似度
        all_item_emb = self.item_embedding.weight[1:]  # (n_items, D) 去掉 padding
        scores = torch.matmul(last_hidden, all_item_emb.T)  # (B, n_items)

        return scores
```

## 3. 训练过程

```python
class SASRecTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-6):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for item_seq, target in dataloader:
            item_seq = item_seq.to(device)   # (B, L)
            target = target.to(device)       # (B,) 下一个物品

            # 预测
            scores = self.model.predict(item_seq)  # (B, n_items)

            # 交叉熵损失
            loss = self.criterion(scores, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, dataloader, device, top_k=[5, 10, 20]):
        self.model.eval()
        metrics = {f'HR@{k}': [] for k in top_k}
        metrics.update({f'NDCG@{k}': [] for k in top_k})

        for item_seq, target in dataloader:
            item_seq = item_seq.to(device)
            target = target.to(device)

            scores = self.model.predict(item_seq)

            for k in top_k:
                _, topk = scores.topk(k, dim=-1)
                for i in range(target.size(0)):
                    if target[i] in topk[i]:
                        metrics[f'HR@{k}'].append(1.0)
                        rank = (topk[i] == target[i]).nonzero()[0].item() + 1
                        metrics[f'NDCG@{k}'].append(1.0 / np.log2(rank + 1))
                    else:
                        metrics[f'HR@{k}'].append(0.0)
                        metrics[f'NDCG@{k}'].append(0.0)

        return {k: np.mean(v) for k, v in metrics.items()}
```

## 4. 因果掩码详解

因果掩码确保每个位置只能看到其前序位置的信息：

```
位置:  0  1  2  3  4
0    [ 0  1  1  1  1 ]   位置 0 只看自己
1    [ 0  0  1  1  1 ]   位置 1 看 0, 1
2    [ 0  0  0  1  1 ]   位置 2 看 0, 1, 2
3    [ 0  0  0  0  1 ]   位置 3 看 0, 1, 2, 3
4    [ 0  0  0  0  0 ]   位置 4 看全部

0 = 可见, 1 = 被屏蔽（填充 -inf 后 softmax 趋近 0）
```

## 5. 数据准备

```python
def prepare_sasrec_data(user_sequences, max_len=50):
    """
    user_sequences: dict {user_id: [item1, item2, ...]} 按时间排序
    返回: (input_sequences, targets) 用于训练
    """
    input_seqs = []
    targets = []

    for user_id, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        # 滑动窗口生成训练样本
        for i in range(1, len(seq)):
            # 输入：截止到位置 i-1 的序列
            start = max(0, i - max_len)
            input_seq = seq[start:i]
            # 目标：位置 i 的物品
            target = seq[i]

            # padding
            padded = [0] * (max_len - len(input_seq)) + input_seq

            input_seqs.append(padded)
            targets.append(target)

    return torch.LongTensor(input_seqs), torch.LongTensor(targets)
```

## 6. SASRec vs BERT4Rec vs GRU4Rec

| 特性 | GRU4Rec | SASRec | BERT4Rec |
|------|---------|--------|----------|
| 编码方式 | RNN (GRU) | 单向自注意力 | 双向自注意力 |
| 并行训练 | 不支持 | 支持 | 支持 |
| 长程依赖 | 受限（梯度消失） | 全局 | 全局 |
| 训练任务 | BPR / CE | 下一个预测 | 掩码预测 |
| 参数量 | 少 | 中等 | 中等 |
| 效果 | 基线 | 强 | 通常最强 |

## 7. 实战技巧

```python
# 1. 负采样加速（物品数大时全 softmax 慢）
class BPRLoss(nn.Module):
    """BPR 损失：正负样本对排序"""
    def forward(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

# 2. 数据增强
def augment_sequence(seq, aug_prob=0.3):
    """随机删除/重排增强"""
    if random.random() < aug_prob:
        # 随机删除 10-30% 的物品
        n_delete = max(1, int(len(seq) * random.uniform(0.1, 0.3)))
        indices = sorted(random.sample(range(len(seq)), len(seq) - n_delete))
        return [seq[i] for i in indices]
    return seq
```

---

**要点总结：**
- SASRec 用单向自注意力建模用户行为序列，预测下一个物品
- 因果掩码保证自回归特性，每个位置只看到前序信息
- Pre-LN（先做 LayerNorm 再做注意力）比 Post-LN 训练更稳定
- SASRec 是序列推荐的经典基线，简洁而有效
