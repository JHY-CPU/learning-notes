# 7_BERT4Rec：序列推荐

## 1. 概述

BERT4Rec (CIKM 2019) 将 BERT 的双向编码和掩码预测思想引入序列推荐，通过预测被掩码的历史物品来学习序列表示。

**核心思想：** 用户行为序列中，每个物品的含义不仅取决于其前序物品，也取决于后续物品。双向 Transformer 能捕捉这种上下文。

```
自回归 (SASRec):    x₁ → x₂ → [MASK] → x₄ → x₅    (只看过去)
双向编码 (BERT4Rec): x₁ → x₂ → [MASK] → x₄ → x₅    (看全部)
```

## 2. 掩码预测任务

在训练时随机掩码一部分历史物品，模型需要从双向上下文中恢复：

```
原始序列:  [item₁, item₂, item₃, item₄, item₅]
掩码序列:  [item₁, [MASK], item₃, [MASK], item₅]
预测目标:  [_, item₂, _, item₄, _]
```

```python
import torch
import torch.nn as nn
import random

class BERT4RecDataset:
    """BERT4Rec 数据处理：随机掩码"""
    def __init__(self, max_len, mask_prob=0.2, mask_token_id=0):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id

    def mask_sequence(self, item_seq):
        """
        item_seq: list of item IDs
        返回: masked_seq, labels, mask_positions
        """
        masked_seq = item_seq.copy()
        labels = [0] * len(item_seq)  # 0 表示不预测
        mask_positions = []

        for i in range(len(item_seq)):
            if random.random() < self.mask_prob:
                mask_positions.append(i)
                labels[i] = item_seq[i]  # 真实标签

                r = random.random()
                if r < 0.8:
                    masked_seq[i] = self.mask_token_id  # 80% [MASK]
                elif r < 0.9:
                    masked_seq[i] = random.randint(1, 10000)  # 10% 随机替换
                # 10% 保持不变

        return masked_seq, labels, mask_positions
```

## 3. BERT4Rec 模型

```python
class BERT4Rec(nn.Module):
    """BERT4Rec: 双向 Transformer 序列推荐"""
    def __init__(self, n_items, max_len=50, embed_dim=64,
                 n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len

        # 物品嵌入 + 位置嵌入
        self.item_embedding = nn.Embedding(n_items + 2, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 输出层：预测被掩码的物品
        self.output_layer = nn.Linear(embed_dim, n_items + 1)

    def forward(self, item_seq):
        """
        item_seq: (B, L) 包含 [MASK]=0 的物品序列
        """
        B, L = item_seq.shape

        # 物品嵌入
        item_emb = self.item_embedding(item_seq)  # (B, L, D)

        # 位置嵌入
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, L)
        pos_emb = self.position_embedding(positions)

        # 合并嵌入
        x = self.layer_norm(self.dropout(item_emb + pos_emb))

        # 注意力掩码：padding 位置不参与注意力
        pad_mask = (item_seq == 0)  # (B, L)

        # Transformer 编码
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)

        # 预测
        logits = self.output_layer(encoded)  # (B, L, n_items+1)
        return logits

    def predict(self, item_seq, target_positions):
        """预测被掩码位置的物品"""
        logits = self.forward(item_seq)  # (B, L, V)
        # 提取掩码位置的预测
        B = item_seq.size(0)
        pred = logits[torch.arange(B).unsqueeze(1),
                      target_positions, :]  # (B, num_mask, V)
        return pred
```

## 4. 训练过程

```python
def train_bert4rec(model, train_sequences, optimizer, device,
                   mask_prob=0.2, n_epochs=50):
    """BERT4Rec 训练循环"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略非掩码位置

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for seq_batch in train_sequences:
            seq_batch = seq_batch.to(device)

            # 随机掩码
            masked_seq, labels = random_mask(seq_batch, mask_prob, device)

            # 前向传播
            logits = model(masked_seq)  # (B, L, V)

            # 计算损失（只在掩码位置）
            loss = criterion(logits.view(-1, logits.size(-1)),
                           labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss/n_batches:.4f}')


def random_mask(item_seq, mask_prob, device):
    """训练时随机掩码"""
    B, L = item_seq.shape
    masked_seq = item_seq.clone()
    labels = torch.zeros_like(item_seq)  # 0 = 不预测

    # 随机决定掩码位置
    mask = torch.rand(B, L, device=device) < mask_prob
    mask &= (item_seq != 0)  # 不掩码 padding

    labels[mask] = item_seq[mask]  # 被掩码位置的标签

    # 掩码处理
    rand = torch.rand(B, L, device=device)
    masked_seq[mask & (rand < 0.8)] = 0   # 80% [MASK]
    masked_seq[mask & (rand >= 0.8) & (rand < 0.9)] = \
        torch.randint(1, item_seq.max()+1, (B, L), device=device)[mask & (rand >= 0.8) & (rand < 0.9)]

    return masked_seq, labels
```

## 5. 推理与评估

```python
@torch.no_grad()
def evaluate_bert4rec(model, test_sequences, top_k=10):
    """评估 BERT4Rec：在最后一个位置放 [MASK]"""
    model.eval()
    hr_list = []
    ndcg_list = []

    for seq, true_next in test_sequences:
        # 在末尾添加 [MASK]
        masked_seq = seq.clone()
        masked_seq[-1] = 0  # [MASK]

        # 预测
        logits = model(masked_seq.unsqueeze(0))  # (1, L, V)
        scores = logits[0, -1, :]  # 最后一个位置的预测

        # Top-K 推荐
        _, topk_items = scores.topk(top_k)
        topk_items = topk_items.cpu().tolist()

        # HR@K
        hr = 1.0 if true_next in topk_items else 0.0
        hr_list.append(hr)

        # NDCG@K
        if true_next in topk_items:
            rank = topk_items.index(true_next) + 1
            ndcg_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_list.append(0.0)

    return np.mean(hr_list), np.mean(ndcg_list)
```

## 6. BERT4Rec vs SASRec

| 特性 | SASRec | BERT4Rec |
|------|--------|----------|
| 编码方向 | 单向（左到右） | 双向 |
| 训练任务 | 下一个物品预测 | 掩码物品预测 |
| 上下文 | 仅过去 | 过去+未来 |
| 掩码策略 | 因果掩码 | 随机掩码 |
| 效果 | 强基线 | 通常更优 |
| 适用场景 | 实时推荐 | 离线推荐 |

---

**要点总结：**
- BERT4Rec 将 BERT 的双向编码引入序列推荐
- 掩码预测任务迫使模型从双向上下文中学习物品表示
- 训练时 80% [MASK] + 10% 随机替换 + 10% 保持不变，防止过拟合
- 推理时在序列末尾添加 [MASK] 来预测下一个物品
