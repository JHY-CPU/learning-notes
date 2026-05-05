# 43_图像检索与哈希学习 (Hash Learning)

## 核心概念

- **图像检索（Image Retrieval）**：给定查询图像，从大规模数据库中找到最相似的图像。核心挑战是在海量图像中高效地进行相似性搜索。
- **哈希学习（Hash Learning / Deep Hashing）**：将图像编码为二进制哈希码（如64位或128位的01序列），使相似图像的哈希码在汉明空间中距离近，不相似图像的哈希码距离远。
- **汉明距离（Hamming Distance）**：两个二进制哈希码之间不同位的数量。汉明距离越小，图像越相似。汉明距离的计算可以通过高效的XOR和PopCount操作实现。
- **哈希码的存储与搜索优势**：二进制哈希码使用位（bit）存储，128位哈希码只需16字节；在汉明空间中使用哈希表或层级搜索，可以在百万/十亿级数据库中实现毫秒级检索。
- **损失函数设计**：成对损失（对比损失、三元组损失）鼓励相似图像对的哈希码接近，不相似对的哈希码远离；量化损失减少哈希码的量化误差（从连续特征到离散二值码的损失）。
- **深度哈希方法演进**：从CNNH（两阶段：特征提取+哈希学习分离）到DSH、HashNet、DPSH（端到端），再到Greedy Hash（贪婪哈希）和CSQ（循环稀疏量化）。

## 数学推导

**哈希学习的目标：**
学习一个哈希函数 $h(x) \in \{-1, +1\}^K$，将图像 $x$ 映射为 $K$ 位的二进制码，使得：
$$
\min \sum_{i,j} \frac{1}{2}(1 - S_{ij}) d_h(h_i, h_j) + \frac{1}{2} S_{ij} \max(0, m - d_h(h_i, h_j))
$$

其中 $S_{ij} = 1$ 表示图像相似，$S_{ij} = 0$ 表示不相似，$d_h$ 是汉明距离，$m$ 是边界。

**连续松弛与量化：**
哈希码的离散性（$h \in \{-1, +1\}^K$）使优化困难。通常使用连续松弛——先用连续的实值特征 $u \in \mathbb{R}^K$ 进行训练：
$$
\mathcal{L} = \mathcal{L}_{similarity}(u) + \lambda \|u\|_{\{-1, +1\}}
$$

第二项是量化损失（促使连续值接近 $\pm 1$）。

**三元组哈希损失：**
$$
\mathcal{L}_{triplet}(a, p, n) = \max(0, \|h_a - h_p\|_H^2 - \|h_a - h_n\|_H^2 + m)
$$

其中 $(a, p, n)$ 是锚点-正例-负例三元组，$\|\cdot\|_H$ 是汉明距离。

**检索评估指标：**
- **mAP（Mean Average Precision）**：对不同查询的AP取平均
- **Precision@K**：返回的前K个结果中相关图像的比例
- **Recall@K**：所有相关图像中，出现在前K个结果中的比例

## 直观理解

哈希学习可以理解为"给每张图像一个简洁的二进制指纹"。这个指纹要满足两条要求：(1) 相似的图像有相似的指纹（汉明距离近）；(2) 指纹尽量短（节省存储）。

想象一下，你需要在一个有100万张照片的图书馆中找到和你手中照片最相似的10张。直接逐一比较（线性搜索）需要比较100万次。但如果每张照片都有一个64位的二进制指纹，你可以先用这些指纹构建哈希表——把指纹相同的照片放在同一个桶里，搜索时只需计算查询图像的指纹，直接找到对应的桶即可（近似最近邻搜索，速度提升数万倍）。

## 代码示例

```python
import torch
import torch.nn as nn

class DeepHashNet(nn.Module):
    """深度哈希网络"""
    def __init__(self, hash_bits=64, pretrained=True):
        super().__init__()
        # 使用ResNet-18作为骨干
        backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # 哈希层
        self.fc = nn.Linear(512, hash_bits)

    def forward(self, x):
        x = self.features(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)        # (B, hash_bits) 连续值
        return x

    def generate_hash(self, x):
        """生成二进制哈希码"""
        code = self.forward(x)
        code = torch.sign(code)  # {-1, +1}
        return code

def hamming_distance(code1, code2):
    """计算两个二进制码之间的汉明距离"""
    # code1, code2: (N, bits) 或 (1, bits) 值为 {-1, +1}
    return 0.5 * (code1.size(1) - (code1 @ code2.T))

def pairwise_hash_loss(codes, labels, margin=2.0):
    """成对哈希损失"""
    B = codes.size(0)
    # 计算相似性矩阵 (基于标签)
    sim_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # 计算汉明距离矩阵
    dist_matrix = hamming_distance(codes, codes)
    
    # 相似对的损失: 距离应小
    similar_loss = sim_matrix * dist_matrix
    # 不相似对的损失: 距离应大
    dissimilar_loss = (1 - sim_matrix) * torch.clamp(margin - dist_matrix, min=0)
    
    loss = (similar_loss.sum() + dissimilar_loss.sum()) / (B * B)
    return loss

# 测试
import torchvision
model = DeepHashNet(hash_bits=64)
x = torch.randn(4, 3, 224, 224)
codes = model(x)
binary_codes = model.generate_hash(x)

print(f"连续哈希码: {codes.shape}")
print(f"二值哈希码: {binary_codes.shape}")
print(f"哈希码值: {binary_codes[0][:8]}... (取值为-1或+1)")

# 测试检索
query = torch.randn(1, 3, 224, 224)
db = torch.randn(10, 3, 224, 224)
q_code = model.generate_hash(query)
db_codes = model.generate_hash(db)
distances = hamming_distance(q_code, db_codes)
print(f"与10个数据库图像的汉明距离: {distances[0]}")
```

## 深度学习关联

- **大规模图像搜索的工业标准**：深度哈希是百度、谷歌、Pinterest等公司大规模图像搜索系统的核心技术。通过将图像编码为64-128位哈希码，可以在数十亿图像库中实现毫秒级搜索。
- **近邻搜索的加速技术**：哈希学习与近似最近邻搜索（ANN）技术互补——前者关注"如何生成好的编码"，后者关注"如何在编码空间中快速搜索"。结合使用（如IVF+PQ、HNSW等索引结构）可以获得最佳搜索性能。
- **跨模态哈希**：哈希学习的原理被扩展到跨模态检索——将图像和文本映射到同一个汉明空间（如Deep Cross-Modal Hashing），实现"以图搜文"或"以文搜图"。
