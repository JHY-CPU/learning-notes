# DIN与序列推荐


## 1. 用户兴趣的多样性与时变性


### 1.1 用户兴趣的特点


- **多样性（Diversity）**
   ：用户的兴趣是多方面的，可能同时喜欢数码产品、运动装备和美食
- **时变性（Temporal Dynamics）**
   ：用户兴趣随时间演化，不同阶段关注不同内容
- **局部性（Local Relevance）**
   ：对于某个候选物品，用户历史行为中只有部分相关


### 1.2 传统方法的局限


传统模型（如DeepFM）将用户历史行为Embedding做简单的**平均池化**（Average Pooling）得到固定长度的用户表示。这会导致：


- 不同兴趣被压缩到同一个向量中，丢失了细粒度信息
- 对于不同的候选物品，用户的表示是相同的，无法体现局部相关性
- 兴趣越多，向量越模糊（兴趣分散问题）


> **Example:** **例子：**
> 用户最近浏览了 [手机壳、篮球鞋、机械键盘、蓝牙耳机]。当候选物品是"键盘托"时，"机械键盘"的行为最重要；当候选物品是"篮球"时，"篮球鞋"的行为最重要。平均池化无法区分这种差异。


## 2. DIN（Deep Interest Network）


### 2.1 核心思想


DIN（阿里, 2018）引入**注意力机制**，针对每个候选物品，对用户历史行为进行加权求和，权重由历史行为与候选物品的相关性决定。


### 2.2 注意力单元（Activation Unit）


$$
Vu(a) = Σi=1n a(ei, ea) · ei = Σi=1n wi · ei
                a(ei, ea) = f(ei, ea, ei-ea, ei⊙ea)
                其中：ei = 第i个历史行为的Embedding，ea = 候选物品Embedding
                f() 为小型DNN（Attention Unit），输出标量权重
$$


### 2.3 DIN的结构特点


- **注意力输入**
   ：[e
   ~i~
   , e
   ~a~
   , e
   ~i~
   -e
   ~a~
   , e
   ~i~
   ⊙e
   ~a~
   ] 四部分拼接
- **权重不归一化**
   ：注意力权重不经过softmax，保留用户的兴趣强度差异
- **用户兴趣表示**
   ：随候选物品动态变化（Local Activation）


> **Note:** **为什么不softmax？**
> DIN保留了用户兴趣的绝对强度。例如用户对数码产品的兴趣本身就很强，不希望被归一化后稀释。实际中用mini-batch内的均值和方差做归一化（类似BatchNorm）。


## 3. DIEN（Deep Interest Evolution Network）


### 3.1 DIEN的动机


DIN只考虑了历史行为与候选物品的相关性，但忽略了**兴趣的演化过程**。用户的兴趣是随时间动态变化的，DIEN通过GRU建模兴趣的时序演化。


### 3.2 兴趣演化网络结构


1. **Embedding层**
   ：将用户行为序列转化为Embedding序列
2. **兴趣提取层（Interest Extractor）**
   ：用GRU对行为序列编码，提取每个时间步的兴趣表示
3. **兴趣演化层（Interest Evolving）**
   ：用AUGRU（Attention-based GRU）建模兴趣随候选物品的演化过程


### 3.3 AUGRU（Attention-based GRU）


$$
ht = (1 - at ⊙ zt) ⊙ ht-1 + at ⊙ zt ⊙ h̃t
                其中 at 为注意力权重，控制每个时间步在演化中的贡献
$$


### 3.4 辅助损失函数


DIEN引入辅助损失（Auxiliary Loss）来监督兴趣提取层的学习：


$$
Laux = -&frac1N Σi=1N-1 [log σ(hi, ei+1) + log(1 - σ(hi, e'i+1))]
                正样本：下一个真实行为 ei+1，负样本：随机采样的行为 e'i+1
$$


$$
Ltotal = Ltarget + α · Laux
$$


## 4. Transformer在序列推荐中的应用


### 4.1 SASRec（Self-Attentive Sequential Recommendation）


SASRec（2018）将Transformer的自注意力机制应用到序列推荐中，用单向注意力（Causal Attention）建模用户行为序列。


- 使用单向Self-Attention，位置t只能看到位置 ≤ t 的行为
- 加入可学习的位置编码
- 使用next-item prediction作为训练目标
- 相比RNN/GRU：可并行训练，能捕获长距离依赖


### 4.2 BERT4Rec


BERT4Rec（2019）将BERT的双向注意力引入序列推荐：


- 使用双向Self-Attention，每个位置可以看到整个序列
- 采用MLM（Masked Language Model）训练策略：随机mask部分物品，预测被mask的物品
- 相比SASRec的单向注意力，双向建模能捕获更丰富的上下文信息


### 4.3 序列推荐模型对比


| 模型 | 序列建模 | 注意力类型 | 训练方式 | 代表论文 |
| --- | --- | --- | --- | --- |
| GRU4Rec | GRU | 无 | next-item | Hidasi, 2016 |
| DIN | 无 | Target Attention | CTR预测 | 阿里, 2018 |
| DIEN | AUGRU | Target Attention | CTR + Aux | 阿里, 2019 |
| SASRec | Self-Attn | 单向Causal | next-item | Kang, 2018 |
| BERT4Rec | Self-Attn | 双向 | Mask预测 | Sun, 2019 |
| S3-Rec | Self-Attn | 双向 | 互信息最大化 | Zhou, 2020 |


## 5. PyTorch代码：DIN实现


```
import torch
import torch.nn as nn

class AttentionUnit(nn.Module):
    """DIN的注意力单元"""

    def __init__(self, embed_dim, hidden_dims=(64, 32)):
        super(AttentionUnit, self).__init__()
        # 输入: [item_emb, candidate_emb, item-candidate, item*candidate]
        input_dim = embed_dim * 4
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.PReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_emb, candidate_emb):
        """
        item_emb: (batch, seq_len, embed_dim) 历史行为
        candidate_emb: (batch, 1, embed_dim) 候选物品
        返回: (batch, seq_len, 1) 注意力权重
        """
        # 扩展candidate维度以便广播
        candidate = candidate_emb.expand_as(item_emb)

        # 四种交互方式
        att_input = torch.cat([
            item_emb,                    # e_i
            candidate,                   # e_a
            item_emb - candidate,        # e_i - e_a
            item_emb * candidate         # e_i * e_a
        ], dim=-1)

        att_weight = self.mlp(att_input)  # (batch, seq_len, 1)
        return att_weight


class DIN(nn.Module):
    """Deep Interest Network"""

    def __init__(self, num_items, embed_dim=32, mlp_dims=(128, 64)):
        super(DIN, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # 注意力单元
        self.attention = AttentionUnit(embed_dim)

        # DNN塔: 拼接用户兴趣+候选+用户特征后输入
        input_dim = embed_dim * 3  # user_interest + candidate + user_profile
        layers = []
        for h_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_hist, candidate_id, mask=None):
        """
        user_hist: (batch, seq_len) 用户历史行为物品ID
        candidate_id: (batch,) 候选物品ID
        mask: (batch, seq_len) 行为序列掩码
        """
        # Embedding
        hist_emb = self.item_embedding(user_hist)        # (batch, seq_len, dim)
        cand_emb = self.item_embedding(candidate_id)     # (batch, dim)
        cand_emb = cand_emb.unsqueeze(1)                 # (batch, 1, dim)

        # 注意力加权
        att_weight = self.attention(hist_emb, cand_emb)  # (batch, seq_len, 1)

        # 应用mask
        if mask is not None:
            att_weight = att_weight.masked_fill(
                mask.unsqueeze(-1) == 0, float('-inf'))
            att_weight = torch.softmax(att_weight, dim=1)

        # 加权求和得到用户兴趣表示
        user_interest = (hist_emb * att_weight).sum(dim=1)  # (batch, dim)

        # 拼接并过DNN
        concat = torch.cat([user_interest, cand_emb.squeeze(1),
                           cand_emb.squeeze(1)], dim=-1)
        output = torch.sigmoid(self.mlp(concat))
        return output.squeeze(-1)


# 使用示例
if __name__ == "__main__":
    model = DIN(num_items=10000, embed_dim=32)
    # 模拟输入
    hist = torch.randint(0, 10000, (4, 20))   # batch=4, seq_len=20
    cand = torch.randint(0, 10000, (4,))       # 候选物品
    mask = (hist != 0).long()                   # padding mask
    output = model(hist, cand, mask)
    print(f"DIN输出: {output}")
```


## 6. 工业实践要点


### 6.1 行为序列的截断策略


- 用户行为序列可能非常长（数千条），但模型需要固定长度输入
- 常用策略：取最近N条（如最近50~200条）
- 更细粒度：按时间窗口截断（最近7天的行为）
- 多兴趣拆分：将长序列按兴趣类别拆分，分别建模


### 6.2 Attention的工程优化


- 注意力计算是DIN/DIEN的性能瓶颈（与序列长度成正比）
- **近似最近邻**
   ：用ANN预筛选最相关的历史行为
- **分组注意力**
   ：将行为按类别分组，组内做注意力
- **缓存机制**
   ：离线预计算注意力权重


### 6.3 选型建议


| 场景 | 推荐模型 | 原因 |
| --- | --- | --- |
| 行为序列短（<20） | DIN | 简单有效，计算开销可控 |
| 行为序列长 | DIEN / SIM | 建模演化，SIM可处理超长序列 |
| 纯序列推荐 | SASRec / BERT4Rec | 不需要候选物品做注意力 |
| 实时性要求高 | SASRec（可离线） | 注意力可预计算 |


<!-- Converted from: 02_DIN与序列推荐.html -->
