# DeepFM与Wide-Deep


## 1. Wide & Deep（Google, 2016）


### 1.1 核心思想


Wide & Deep模型由Google在2016年提出，用于Google Play的App推荐。其核心思想是结合**记忆能力**（Memorization）和**泛化能力**（Generalization）：


- **Wide部分**
   （线性模型）：记忆历史数据中频繁出现的特征组合，直接利用共现信息
- **Deep部分**
   （DNN）：通过低维稠密Embedding学习之前未见过的特征组合，具有更好的泛化性


### 1.2 模型结构


$$
P(y=1|x) = σ(wwideT[x, φ(x)] + adeep(l) + b)
                Wide部分：ywide = wT[x, φ(x)] + b
                Deep部分：a(l+1) = f(W(l)a(l) + b(l))
$$


其中φ(x)为交叉特征变换（如AND(user_installed_app=QQ, impression_app=微信)），Wide和Deep的输出拼接后经过sigmoid得到最终预测概率。


### 1.3 Wide与Deep的分工


| 组件 | 功能 | 模型 | 特征 |
| --- | --- | --- | --- |
| Wide | 记忆 | 线性模型（LR） | 稀疏特征 + 交叉特征 |
| Deep | 泛化 | 多层DNN | 稠密Embedding |


> **Note:** **Google的工程实践：**
> 线上Serving时，Wide部分直接查表计算，Deep部分前向传播，两部分加权求和得到最终结果。整个系统实现了在线学习（Online Learning），每隔几分钟更新一次模型参数。


## 2. DeepFM（华为, 2017）


### 2.1 动机与改进


Wide & Deep的Wide部分需要手动设计交叉特征（φ(x)），DeepFM用**FM层替代Wide部分**，自动学习二阶特征交叉，无需人工特征工程。


### 2.2 模型结构


$$
y = σ(yFM + yDNN)
                FM部分：yFM = <w, x> + ΣiΣj>i<vi, vj>xixj
                DNN部分：yDNN = f(W(l)a(l) + b(l))
$$


### 2.3 DeepFM的核心创新：共享Embedding


FM部分和DNN部分**共享同一个Embedding层**，这是DeepFM的关键设计：


- 每个特征的Embedding向量同时服务于FM的二阶交叉和DNN的高阶交叉
- 低频特征的Embedding也能得到充分训练（因为FM和DNN联合训练）
- 相比Wide & Deep不需要手动设计交叉特征
- 端到端训练，无需预训练


### 2.4 DeepFM vs Wide & Deep


| 维度 | Wide & Deep | DeepFM |
| --- | --- | --- |
| 低阶交叉 | 手动设计交叉特征 | FM自动学习二阶交叉 |
| Embedding | Wide和Deep不共享 | FM和DNN共享Embedding |
| 特征工程 | 需要（Wide部分） | 无需手动交叉特征 |
| 训练方式 | 需要预训练Deep部分 | 端到端训练 |
| 工业应用 | Google Play | 华为应用市场 |


## 3. DCN（Deep & Cross Network）


### 3.1 Cross Network


DCN用**Cross Network**替代FM/Wide部分，显式地对特征进行有限阶交叉：


$$
xl+1 = x0 ⊙ (Wlxl + bl) + xl
                其中⊙为逐元素乘法（Hadamard积），x0为原始输入特征
$$


### 3.2 DCN-V2


DCN-V2引入了MoE（Mixture of Experts）结构，每个Cross层使用多个专家：


$$
xl+1 = Σk=1K Gk(xl) · Expertk(xl) + xl
$$


### 3.3 各模型交叉能力对比


| 模型 | 交叉方式 | 交叉阶数 | 复杂度 |
| --- | --- | --- | --- |
| LR | 无交叉（线性） | 1阶 | O(n) |
| FM | 隐向量内积 | 2阶 | O(kn) |
| DNN | 隐式高阶交叉 | L层 = L阶 | O(Σn~l~n~l+1~) |
| Cross Net | 显式交叉 | L层 = L+1阶 | O(Ldn) |


## 4. DIN（Deep Interest Network）简介


### 4.1 从特征交叉到兴趣建模


上述模型（Wide & Deep、DeepFM、DCN）主要关注特征交叉，而DIN（阿里, 2018）将注意力转向**用户兴趣建模**：不同的历史行为对当前候选物品的重要性不同。


### 4.2 Attention机制


$$
Vu = f({ei}) = Σi a(ei, ea) · ei
                a(ei, ea)：注意力权重，ea为候选物品Embedding
                Vu：加权后的用户兴趣表示
$$


详见下一章 "DIN与序列推荐" 的深入讲解。


## 5. 深度推荐模型结构对比


| 模型 | 年份 | 低阶特征 | 高阶特征 | 核心创新 |
| --- | --- | --- | --- | --- |
| LR | - | 线性组合 | 无 | 基线模型 |
| FM | 2010 | 隐向量二阶交叉 | 无 | 自动特征交叉 |
| Wide & Deep | 2016 | 交叉特征（手工） | DNN | 记忆+泛化 |
| DeepFM | 2017 | FM自动交叉 | DNN | 共享Embedding |
| DCN | 2017 | Cross Net显式交叉 | DNN | 有限阶显式交叉 |
| DIN | 2018 | - | DNN+Attention | 兴趣加权 |
| xDeepFM | 2018 | CIN压缩交互 | DNN | 向量级交叉 |


> **Important:** **工业选型建议：**
> 在CTR预估任务中，DeepFM通常是首选的基线模型，因为它简单有效且无需手工特征工程。如果用户行为序列丰富，应考虑DIN/DIEN。如果是多目标场景，则需要结合MMOE等多任务框架。


## 6. PyTorch代码：实现DeepFM


```
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """DeepFM模型实现"""

    def __init__(self, field_dims, embed_dim=8, mlp_dims=(128, 64),
                 dropout=0.2):
        """
        field_dims: list, 每个field的特征数
        embed_dim: Embedding维度
        mlp_dims: DNN隐层维度
        """
        super(DeepFM, self).__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # Embedding层（FM和DNN共享）
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        # 偏移量，用于计算每个field在embedding表中的起始位置
        self.offsets = torch.cumsum(
            torch.tensor([0] + field_dims[:-1]), dim=0)

        # FM一阶：每个特征一个权重
        self.fm_first_order = nn.Embedding(sum(field_dims), 1)

        # DNN部分
        input_dim = self.num_fields * embed_dim
        layers = []
        for mlp_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, mlp_dim))
            layers.append(nn.BatchNorm1d(mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = mlp_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, num_fields) 每个field的特征索引
        """
        # 计算偏移
        x = x + self.offsets.unsqueeze(0).to(x.device)

        # ===== FM一阶 =====
        fm_first = self.fm_first_order(x).sum(dim=1)  # (batch, 1)

        # ===== FM二阶 =====
        embed = self.embedding(x)  # (batch, num_fields, embed_dim)
        # 平方和 - 和的平方
        square_of_sum = embed.sum(dim=1).pow(2)        # (batch, embed_dim)
        sum_of_square = embed.pow(2).sum(dim=1)        # (batch, embed_dim)
        fm_second = 0.5 * (square_of_sum - sum_of_square).sum(
            dim=1, keepdim=True)                       # (batch, 1)

        # ===== DNN =====
        dnn_input = embed.view(embed.size(0), -1)      # (batch, num_f*emb)
        dnn_output = self.mlp(dnn_input)                # (batch, 1)

        # ===== 合并 =====
        output = torch.sigmoid(fm_first + fm_second + dnn_output)
        return output.squeeze(1)


# 使用示例
if __name__ == "__main__":
    # 3个field，每个field分别有100, 50, 20个特征
    field_dims = [100, 50, 20]
    model = DeepFM(field_dims, embed_dim=8, mlp_dims=(128, 64))

    # 模拟输入：(batch=4, num_fields=3)
    x = torch.randint(0, 50, (4, 3))
    output = model(x)
    print(f"模型输出: {output}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```


<!-- Converted from: 01_DeepFM与Wide-Deep.html -->
