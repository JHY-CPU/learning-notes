# 44_零样本学习 (Zero-Shot Learning) 在 CV 中的应用

## 核心概念

- **零样本学习（Zero-Shot Learning, ZSL）**：模型在测试时能够识别训练阶段从未见过的类别。这通过利用类别的辅助信息（如语义属性、词向量、文本描述）来实现"知识迁移"。
- **语义属性（Semantic Attributes）**：描述类别共享的视觉属性，如"有羽毛"、"有翅膀"、"会飞"等。属性构成一个中间的语义层，连接已见类和未见类。
- **经典ZSL设置**：训练集和测试集的类别是不相交的（disjoint）。例如，训练集包含"马、老虎、鸭子"，测试集包含"斑马、熊猫、天鹅"。模型需要在测试时识别这些未见过的新类别。
- **广义零样本学习（Generalized ZSL, GZSL）**：测试集同时包含已见类和未见类。模型需要区分所有类别，而不仅仅是新类别（更具挑战性）。
- **视觉-语义嵌入（Visual-Semantic Embedding）**：将视觉特征和语义描述映射到同一嵌入空间，通过计算视觉特征与类别语义原型之间的相似性进行分类。
- **直推式零样本学习（Transductive ZSL）**：训练时可以利用未标注的测试集图像（只利用其视觉特征，不使用标签），通过聚类或流形正则化辅助学习。

## 数学推导

**零样本学习的基本框架：**

给定已见类 $S$ 的训练数据 $\{(x_i, y_i)\}_{i=1}^N$，其中 $y_i \in S$。对每个类别 $c$ 有语义向量 $a_c \in \mathbb{R}^D$（属性或词向量）。

目标：学习映射函数 $f: \mathcal{X} \to \mathcal{Y}_{all}$，其中 $\mathcal{Y}_{all} = S \cup U$（已见类 + 未见类）。

**嵌入方法（Embedding-based ZSL）：**
学习视觉特征到语义空间的投影矩阵 $W$：
$$
\min_W \frac{1}{N} \sum_{i=1}^N \|\phi(x_i) - W a_{y_i}\|_2^2 + \lambda \|W\|_F^2
$$

其中 $\phi(x)$ 是CNN提取的视觉特征。

测试时，对未见类图像 $x_u$，选择最接近的类别原型：
$$
\hat{y} = \arg\min_{c \in U} \|\phi(x_u) - W a_c\|_2
$$

**兼容性函数（Compatibility Function）：**
$$
F(x, c) = \phi(x)^T W a_c
$$

预测类别为兼容性最高的类别：
$$
\hat{y} = \arg\max_{c \in \mathcal{Y}_{all}} F(x, c)
$$

**自编码器方法（如GAZSL）：**
使用自编码器重建语义向量，在未见类上产生更好的泛化：
$$
\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{class} + \mathcal{L}_{align}
$$

## 直观理解

零样本学习模仿了人类的"举一反三"能力。如果告诉你"斑马像马但有黑白条纹"，当你第一次看到斑马时，即使从未见过，也能根据"像马"（已知类别的视觉特征）和"黑白条纹"（属性的组合）来识别它。

在ZSL中，"属性"（如"有条纹"、"有四条腿"）扮演了"桥梁"的角色。模型在训练阶段学会识别这些属性（从已见类如老虎、斑马的图片中学到"条纹"模式），在测试阶段通过未见类的属性组合来识别新类别。"有条纹+四条腿+像马"→ 斑马。

广义零样本学习（GZSL）之所以更难，是因为模型有"偏好已见类"的倾向——就像让一个从未见过熊猫的人找熊猫，他会更倾向于把他见过的狗或熊误判为熊猫，因为他更熟悉"已见类"的特征分布。

## 代码示例

```python
import torch
import torch.nn as nn

class ZeroShotModel(nn.Module):
    """零样本学习模型: 视觉-语义嵌入"""
    def __init__(self, visual_dim=2048, attr_dim=85, embedding_dim=1024):
        super().__init__()
        # 视觉特征投影
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # 语义特征投影
        self.semantic_proj = nn.Sequential(
            nn.Linear(attr_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, visual_feat, semantic_vec):
        v = self.visual_proj(visual_feat)
        s = self.semantic_proj(semantic_vec)
        # 归一化
        v = nn.functional.normalize(v, dim=1)
        s = nn.functional.normalize(s, dim=1)
        # 兼容性得分 (余弦相似度)
        compatibility = v @ s.T
        return compatibility

# 属性映射 (示例: CUB-200鸟类数据集的部分属性)
# 每个类别有85维的属性向量 (如: 有红色、有蓝色、有长喙、翅膀颜色等)
class_attributes = {
    'cardinal':  torch.tensor([1, 0, 1, 0, 0.8, 0, 0.2, ...]),  # 红衣主教鸟
    'blue_jay':  torch.tensor([0, 1, 0, 0.8, 0, 0.3, 0, ...]),   # 蓝松鸦
    'sparrow':   torch.tensor([0, 0, 0, 0, 0.5, 0.6, 0.7, ...]), # 麻雀
}
# 注意: 实际属性维度为85, 这里示意

model = ZeroShotModel()

# 训练: 已见类 (马、老虎、鸭子)
visual_feat = torch.randn(4, 2048)  # 从预训练CNN提取的视觉特征
semantic_vec = torch.randn(4, 85)   # 对应的语义属性向量
compat = model(visual_feat, semantic_vec)
print(f"兼容性矩阵: {compat.shape}")

# 测试: 未见类 (斑马)
zebra_visual = torch.randn(1, 2048)
# 计算与所有类别原型的兼容性 (已见+未见)
all_semantics = torch.randn(50, 85)  # 包含已见类和未见类的语义向量
zebra_compat = model(zebra_visual, all_semantics)
pred_class = zebra_compat.argmax(dim=1)
print(f"预测类别索引: {pred_class.item()}")

# 评估指标: 在未见类上的 Top-1 准确率
def compute_zsl_accuracy(visual_feats, semantics, labels):
    """计算零样本学习的准确率"""
    model.eval()
    with torch.no_grad():
        compat = model(visual_feats, semantics)
        preds = compat.argmax(dim=1)
        accuracy = (preds == labels).float().mean()
    return accuracy

print("\n零样本学习的关键挑战:")
print("- 域偏移: 已见类和未见类的视觉特征分布可能存在偏移")
print("- 枢纽点问题: 某些语义向量频繁成为最近邻")
print("- GZSL的偏见: 模型倾向于预测已见类")
```

## 深度学习关联

- **开放词汇视觉识别的基础**：零样本学习技术是开放词汇目标检测（OVR-CNN、GLIP、Grounding DINO）和开放词汇语义分割（OpenScene、OVSeg）的核心技术。这些模型使用CLIP等视觉-语言模型的嵌入来识别训练中未定义的类别。
- **基础模型（Foundation Model）中的ZSL能力**：CLIP、SAM等基础模型展现出强大的零样本泛化能力——通过大规模的图文预训练，它们可以在未见过的任务和类别上直接工作，无需收集任何特定任务的训练数据。
- **从ZSL到Few-Shot的延伸**：零样本学习自然地扩展到少样本学习（Few-Shot Learning）——当只有少量标注样本可用时，结合语义属性或类别原型的先验知识，可以显著提升少样本学习的效果。
