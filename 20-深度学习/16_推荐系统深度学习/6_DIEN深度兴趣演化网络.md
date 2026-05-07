# 6_DIEN 深度兴趣演化网络

## 1. 概述

DIEN (Deep Interest Evolution Network, Alibaba, 2019) 在 DIN 基础上进一步建模用户兴趣的**时序演化**过程。核心创新：
1. **兴趣提取层**：用 GRU 从行为序列中提取兴趣状态
2. **兴趣演化层**：AUGRU (Attention-based GRU) 在兴趣演化过程中引入注意力

```
用户行为序列: 点击球鞋 → 浏览键盘 → 购买鼠标 → 观看耳机评测
                ↓          ↓          ↓           ↓
兴趣状态:    兴趣₁ ──→  兴趣₂ ──→  兴趣₃ ──→  兴趣₄
                (GRU 兴趣提取)
                        ↓
            AUGRU: 根据候选物品选择性地跟踪兴趣演化
                        ↓
                最终兴趣表示 → 预测
```

## 2. 兴趣提取：GRU

GRU 从用户行为序列中提取隐式兴趣状态：

$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

```python
import torch
import torch.nn as nn

class InterestExtractor(nn.Module):
    """兴趣提取层：用 GRU 提取兴趣状态"""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, behavior_embeds, mask=None):
        """
        behavior_embeds: (B, L, D) 行为嵌入序列
        mask: (B, L) 有效位置掩码
        """
        gru_out, _ = self.gru(behavior_embeds)  # (B, L, H)
        return gru_out  # 每个时间步的兴趣状态
```

## 3. 辅助损失 (Auxiliary Loss)

为每个兴趣状态学习有意义的表示，DIEN 设计了辅助损失——预测下一个行为：

$$L_{aux} = -\frac{1}{N}\sum_{t=1}^{N-1}\left[\log\sigma(h_t^i, e_b^{i[t+1]}) + \log(1-\sigma(h_t^i, \hat{e}_b^{i[t+1]}))\right]$$

- 正样本：实际下一个行为 $e_b^{i[t+1]}$
- 负样本：随机采样的行为 $\hat{e}_b^{i[t+1]}$

```python
class AuxiliaryLoss(nn.Module):
    """辅助损失：预测下一个行为"""
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim + embed_dim, 1)

    def forward(self, gru_outputs, pos_next_behavior, neg_next_behavior):
        """
        gru_outputs: (B, L-1, H) GRU 隐状态
        pos_next_behavior: (B, L-1, D) 正样本（下一个真实行为）
        neg_next_behavior: (B, L-1, D) 负样本（随机采样行为）
        """
        # 正样本得分
        pos_input = torch.cat([gru_outputs, pos_next_behavior], dim=-1)
        pos_score = torch.sigmoid(self.W(pos_input))

        # 负样本得分
        neg_input = torch.cat([gru_outputs, neg_next_behavior], dim=-1)
        neg_score = torch.sigmoid(self.W(neg_input))

        # 二元交叉熵
        loss = -torch.mean(
            torch.log(pos_score + 1e-8) + torch.log(1 - neg_score + 1e-8)
        )
        return loss
```

## 4. AUGRU: 注意力更新门 GRU

AUGRU 将注意力权重融入 GRU 的更新门，控制每个时间步对最终兴趣表示的贡献：

$$\tilde{u}_t' = a_t \cdot u_t$$
$$h_t = (1 - \tilde{u}_t') \odot h_{t-1} + \tilde{u}_t' \odot \tilde{h}_t$$

其中 $a_t$ 是基于候选物品的注意力权重。

```python
class AUGRUCell(nn.Module):
    """AUGRU 单元：注意力更新门"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 门控参数
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, attn_weight):
        """
        x_t: (B, D) 当前输入
        h_prev: (B, H) 上一隐状态
        attn_weight: (B, 1) 注意力权重
        """
        combined = torch.cat([h_prev, x_t], dim=-1)

        # 重置门
        r_t = torch.sigmoid(self.W_r(combined))

        # 更新门（被注意力权重调制）
        z_t = torch.sigmoid(self.W_z(combined))
        z_t = attn_weight * z_t  # 关键：注意力调制更新门

        # 候选隐状态
        combined_reset = torch.cat([r_t * h_prev, x_t], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # 最终隐状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class AUGRU(nn.Module):
    """AUGRU 序列处理"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = AUGRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sequence, attention_weights, init_hidden=None):
        """
        sequence: (B, L, D)
        attention_weights: (B, L) 每个时间步的注意力权重
        """
        B, L, D = sequence.shape
        h = init_hidden if init_hidden is not None else \
            torch.zeros(B, self.hidden_dim, device=sequence.device)

        outputs = []
        for t in range(L):
            h = self.cell(sequence[:, t, :], h, attention_weights[:, t:t+1])
            outputs.append(h)

        return torch.stack(outputs, dim=1), h  # (B, L, H), (B, H)
```

## 5. DIEN 完整模型

```python
class DIEN(nn.Module):
    """Deep Interest Evolution Network"""
    def __init__(self, item_vocab_size, embed_dim=64, hidden_dim=36,
                 use_aux_loss=True):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.hidden_dim = hidden_dim

        # 行为嵌入
        self.item_embedding = nn.Embedding(item_vocab_size, embed_dim)

        # 兴趣提取层
        self.interest_extractor = InterestExtractor(embed_dim, hidden_dim)

        # 辅助损失
        if use_aux_loss:
            self.aux_loss = AuxiliaryLoss(hidden_dim, embed_dim)

        # 兴趣演化层
        self.augru = AUGRU(hidden_dim, hidden_dim)

        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, 80),
            nn.PReLU(),
            nn.Linear(80, 40),
            nn.PReLU(),
            nn.Linear(40, 1)
        )

        # 预测层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 80),
            nn.PReLU(),
            nn.Linear(80, 1)
        )

    def forward(self, behavior_seq, candidate_id, mask, pos_next=None, neg_next=None):
        """
        behavior_seq: (B, L) 历史行为物品 ID
        candidate_id: (B,) 候选物品 ID
        mask: (B, L) 有效行为掩码
        pos_next: (B, L-1) 辅助损失正样本
        neg_next: (B, L-1) 辅助损失负样本
        """
        # 1. 行为嵌入
        behavior_embeds = self.item_embedding(behavior_seq)  # (B, L, D)
        candidate_embed = self.item_embedding(candidate_id)  # (B, D)

        # 2. 兴趣提取
        interest_states = self.interest_extractor(behavior_embeds)  # (B, L, H)

        # 3. 辅助损失
        aux_loss = 0
        if self.use_aux_loss and pos_next is not None:
            pos_embeds = self.item_embedding(pos_next)  # (B, L-1, D)
            neg_embeds = self.item_embedding(neg_next)
            aux_loss = self.aux_loss(interest_states[:, :-1, :], pos_embeds, neg_embeds)

        # 4. 计算注意力权重
        candidate_expanded = candidate_embed.unsqueeze(1).expand(-1, interest_states.size(1), -1)
        attn_input = torch.cat([
            interest_states, candidate_expanded,
            interest_states - candidate_expanded,
            interest_states * candidate_expanded
        ], dim=-1)

        attn_scores = self.attention_net(attn_input).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, L)

        # 5. 兴趣演化 (AUGRU)
        _, final_interest = self.augru(interest_states, attn_weights)

        # 6. 预测
        combined = torch.cat([final_interest, candidate_embed], dim=-1)
        output = torch.sigmoid(self.fc(combined))

        return output.squeeze(-1), aux_loss, attn_weights
```

## 6. DIN vs DIEN

| 特性 | DIN | DIEN |
|------|-----|------|
| 兴趣表示 | 注意力直接加权 | GRU 序列建模 |
| 时序关系 | 不建模 | 通过 GRU 建模 |
| 兴趣演化 | 无 | AUGRU 演化建模 |
| 辅助损失 | 无 | 预测下一个行为 |
| 模型复杂度 | 较低 | 较高 |
| 推理速度 | 较快 | 较慢 |

---

**要点总结：**
- DIEN 通过 GRU 从行为序列中提取兴趣状态，比直接嵌入池化更丰富
- AUGRU 将注意力机制融入 GRU 的更新门，根据候选物品选择性跟踪兴趣演化
- 辅助损失迫使中间兴趣状态学习有意义的表示，不参与推理但提升训练效果
- DIEN 适合行为序列较长、兴趣演化明显的场景（如电商）
