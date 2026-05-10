# CTR预估模型演进 - 搜索推荐广告算法


## 1. CTR预估模型演进全景


| 模型 | 年份 | 核心创新 | 特征交互方式 |
| --- | --- | --- | --- |
| LR | 传统 | 线性模型，依赖人工特征工程 | 无自动交叉 |
| GBDT+LR | 2014 | Facebook，树模型做特征转换 | GBDT叶子节点编码 |
| FM | 2010 | 二阶特征交叉的隐向量内积 | 隐向量内积 |
| FFM | 2015 | 场感知因子机，每字段一个隐向量 | 场感知内积 |
| Wide & Deep | 2016 | Google，记忆+泛化 | Wide线性 + Deep DNN |
| DeepFM | 2017 | FM替代Wide部分 | FM + DNN并行 |
| DIN | 2018 | 阿里，用户兴趣建模 | 注意力机制 |
| ESMM | 2018 | 阿里，CVR多任务学习 | CTR和CVR联合建模 |
| DCN V2 | 2020 | 交叉网络高效特征交叉 | Cross Network |
| xDeepFM | 2018 | CIN显式高阶交叉 | Compressed Interaction Network |


## 2. LR与GBDT+LR


### 逻辑回归（LR）


$$
pCTR = σ(wᵀx + b) = 1 / (1 + exp(-(wᵀx + b)))
$$


优点：可解释性强、训练快、易于上线；缺点：无法自动捕获特征交叉，依赖人工特征工程。


### GBDT+LR（Facebook, 2014）


用GBDT的叶子节点编号作为新的特征，输入到LR中进行CTR预估。


> **Example:** **流程：**
> 原始特征 → GBDT训练 → 每个样本落在各棵树的叶子节点 → one-hot编码叶子节点 → LR训练
>
>
> 例如：3棵树，每棵8个叶子 → 产生24维one-hot特征


## 3. FM与FFM


### FM（Factorization Machine）


$$
y(x) = w₀ + Σ wᵢxᵢ + Σ Σ <vᵢ, vⱼ> xᵢxⱼ
                其中 vᵢ 为特征i的k维隐向量，<vᵢ, vⱼ> = Σ(vᵢf × vⱼf)
$$


- 通过隐向量内积自动学习特征交叉，不需要显式枚举交叉特征
- 即使特征i和j没有同时出现过，也能通过隐向量计算交叉权重
- 时间复杂度O(nk)，可以线性时间计算


### FFM（Field-aware Factorization Machine）


FM的改进：每个特征针对不同字段使用不同的隐向量。


$$
y(x) = w₀ + Σ wᵢxᵢ + Σ Σ <vᵢ,fⱼ, vⱼ,fᵢ> xᵢxⱼ
                特征i针对字段fⱼ使用隐向量vᵢ,fⱼ
$$


## 4. Wide & Deep与DeepFM


### Wide & Deep（Google, 2016）


Wide部分（LR）负责记忆已有特征组合，Deep部分（DNN）负责泛化到新的特征组合。


$$
pCTR = σ(σ_wide(x) + σ_deep(x))
$$


- Wide：线性模型，直接记忆已有的特征组合模式
- Deep：多层DNN，通过Embedding+MLP学习高阶特征交互
- 联合训练：Wide和Deep的输出相加后通过sigmoid


### DeepFM（华为, 2017）


用FM替代Wide部分，共享Embedding层，FM和DNN并行。


- FM部分：自动二阶特征交叉，无需手工特征
- Deep部分：高阶特征交叉
- 共享Embedding：减少参数，FM和DNN共享特征表示


## 5. DIN与ESMM


### DIN（Deep Interest Network, 阿里 2018）


针对用户历史行为序列，使用注意力机制对不同历史行为赋予不同权重，建模用户兴趣的多样性。


$$
用户兴趣表示 = Σ aᵢ × eᵢ
                其中 aᵢ = attention(eᵢ, e_target) 为目标注意力权重
$$


> **Note:** DIN的核心洞察：用户的兴趣是多样的，面对不同候选商品，用户的相关历史行为应该不同。例如，推荐手机时关注用户买电子产品的历史，推荐衣服时关注用户买服饰的历史。


### ESMM（Entire Space Multi-Task Model, 阿里 2018）


解决CVR预估的样本选择偏差问题，联合建模CTR和CVR。


$$
pCTCVR = pCTR × pCVR
                pCTR = f_ctr(用户, 物品)
                pCVR = f_cvr(用户, 物品)
                pCTCVR = f_ctcvr(用户, 物品) = pCTR × pCVR
$$


- 问题：CVR训练数据仅包含点击样本（样本选择偏差）
- 解决：在整个曝光空间上训练，CTCVR任务和CTR任务共享Embedding
- 优势：消除样本选择偏差，可利用全量曝光数据


## 6. 特征交互方法：DCN与xDeepFM


### DCN V2（Deep & Cross Network）


用Cross Network显式捕获有界阶数的特征交叉，与Deep部分并行。


```
import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    """DCN V2 Cross层"""
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            # x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
            x = x0 * (x @ self.weights[i] + self.biases[i]) + x
        return x

class DCNV2(nn.Module):
    def __init__(self, feature_dim, hidden_dims=[256, 128], cross_layers=3):
        super().__init__()
        # Cross Network
        self.cross = CrossNetwork(feature_dim, cross_layers)
        # Deep Network
        deep_layers = []
        input_dim = feature_dim
        for h in hidden_dims:
            deep_layers.extend([nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            input_dim = h
        self.deep = nn.Sequential(*deep_layers)
        # 输出层
        self.output = nn.Linear(feature_dim + hidden_dims[-1], 1)

    def forward(self, x):
        cross_out = self.cross(x)
        deep_out = self.deep(x)
        combined = torch.cat([cross_out, deep_out], dim=1)
        return torch.sigmoid(self.output(combined))
```


### xDeepFM


用CIN（Compressed Interaction Network）显式建模向量级的高阶特征交叉。


- CIN在向量级别而非bit级别进行交叉，保留特征的结构信息
- 可以显式控制交叉阶数（1阶、2阶、...、k阶）
- 与DNN结合，同时捕获显式和隐式特征交互


| 特征交互方法 | 交叉阶数 | 交互粒度 | 是否显式 |
| --- | --- | --- | --- |
| Cross Network (DCN) | 有界 | Bit级 | 显式 |
| CIN (xDeepFM) | 有界 | 向量级 | 显式 |
| DNN | 无界 | Bit级 | 隐式 |
| FM | 二阶 | 向量级 | 显式 |


## 总结


- CTR预估从LR的线性模型发展到深度学习模型，核心是特征交互方式的进化
- FM通过隐向量内积自动学习二阶特征交叉，是深度CTR模型的基石
- Wide & Deep开创了记忆+泛化的双路结构，DeepFM用FM替代Wide部分
- DIN引入注意力机制建模用户兴趣的多样性
- ESMM通过多任务学习解决CVR预估的样本选择偏差问题
- DCN和xDeepFM分别从bit级和向量级显式建模高阶特征交互


<!-- Converted from: 02_CTR预估模型演进.html -->
