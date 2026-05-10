# CNN架构演进 — 从LeNet到Vision Transformer

*从LeNet到Vision Transformer — 卷积神经网络的发展历程与核心创新*


**卷积神经网络（Convolutional Neural Network, CNN）**是深度学习在计算机视觉领域取得突破的核心架构。自1998年LeNet提出以来，CNN经历了从浅层到深层、从手工设计到自动搜索、从纯卷积到与Transformer融合的演进历程。


CNN的三大核心思想：


- **局部连接（Local Connectivity）**
   ：每个神经元只与输入的局部区域连接，捕获空间局部特征
- **权值共享（Weight Sharing）**
   ：同一卷积核在整张特征图上共享参数，大幅减少参数量
- **层次化特征提取**
   ：浅层提取边缘、纹理等低级特征，深层提取语义级别的高级特征


### CNN基本组件


| 组件 | 功能 | 典型参数 |
| --- | --- | --- |
| 卷积层（Conv） | 提取局部特征 | kernel_size=3, stride=1, padding=1 |
| 池化层（Pooling） | 降维、增加平移不变性 | MaxPool 2×2, stride=2 |
| 批归一化（BN） | 加速训练、稳定梯度 | 对每个mini-batch标准化 |
| 激活函数 | 引入非线性 | ReLU, GELU, Swish |
| 全连接层（FC） | 分类/回归输出 | 通常位于网络末端 |


### 3.1 各架构参数量与性能


| 模型 | 年份 | 层数 | 参数量 | ImageNet Top-1 | 核心创新 |
| --- | --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | 7 | 60K | — | 卷积+池化基本结构 |
| AlexNet | 2012 | 8 | 60M | 63.3% | ReLU, Dropout, GPU训练 |
| VGG-16 | 2014 | 16 | 138M | 74.4% | 小卷积核堆叠 |
| GoogLeNet | 2014 | 22 | 5M | 74.8% | Inception多尺度 |
| ResNet-152 | 2015 | 152 | 60M | 77.8% | 残差连接 |
| DenseNet-264 | 2017 | 264 | 33M | 77.9% | 密集连接 |
| EfficientNet-B7 | 2019 | — | 66M | 84.3% | 复合缩放 |
| ViT-L/16 | 2020 | 24 | 307M | 85.3% | 纯注意力机制 |


### 4.1 为什么需要跳跃连接


深层网络面临的两个核心问题：


1. **梯度消失/爆炸**
   ：梯度在反向传播过程中逐层衰减或放大
2. **网络退化**
   ：更深的网络反而性能更差（不是过拟合，是优化困难）


跳跃连接使梯度可以绕过非线性层直接回传，有效缓解上述问题。


### 4.2 跳跃连接的类型


| 类型 | 公式 | 使用场景 |
| --- | --- | --- |
| 恒等连接（Identity） | y = F(x) + x | ResNet（维度相同时） |
| 投影连接（Projection） | y = F(x) + Wx | ResNet（维度变化时，1×1卷积） |
| 密集连接（Dense） | y = F([x₁, x₂, ..., xₙ₋₁]) | DenseNet |
| U-Net跳跃 | y = F(x) + UpSample(skip) | U-Net解码器 |
| Highway连接 | y = F(x)·T(x) + x·(1-T(x)) | Highway Network |


> **Note:** **ResNet-v2改进（Pre-activation）：**
> 将BN和ReLU移到卷积之前，形成 BN→ReLU→Conv 的顺序。
>
>
> 原始：y = F(ReLU(BN(x))) + x
>
>
> 改进：y = F(BN(ReLU(x))) + x　→　恒等映射路径完全畅通


### 5.1 原理


标准卷积将空间维度和通道维度混合计算，而深度可分离卷积将其分解为两个独立步骤：


1. **深度卷积（Depthwise Conv）**
   ：每个通道独立进行空间卷积，只提取空间特征
2. **逐点卷积（Pointwise Conv）**
   ：1×1卷积跨通道组合特征


### 5.2 计算量对比


$$
标准卷积: D_K² · M · N · D_F²
        深度可分离: D_K² · M · D_F² + M · N · D_F²
        压缩比: 1/N + 1/D_K²　（当 D_K=3 时约 1/9）
$$


其中 D_K 为卷积核大小，M为输入通道数，N为输出通道数，D_F为特征图大小。


### 5.3 应用


- **MobileNet v1/v2/v3**
   ：移动端轻量级网络，基于深度可分离卷积
- **Xception**
   ：Extreme Inception，用深度可分离卷积替代Inception模块
- **EfficientNet**
   ：MBConv模块（Mobile Inverted Bottleneck）使用深度可分离卷积


```
MBConv模块结构:
Input → 1×1 Conv (expand) → DepthwiseConv3×3 → SE Attention → 1×1 Conv (project) → + Input
其中SE(Squeeze-and-Excitation): GlobalAvgPool → FC → ReLU → FC → Sigmoid → Scale
```


### 6.1 通道注意力 — SENet (Squeeze-and-Excitation)


通过学习每个通道的重要性权重，自适应地校准通道特征响应。


$$
z_c = (1/H×W) Σᵢ Σⱼ x_c(i,j)　→　s = σ(W₂·ReLU(W₁·z))　→　x̃_c = s_c · x_c
$$


SE模块仅增加约2%的参数和0.5%的计算量，却能提升约1%的ImageNet准确率。


### 6.2 其他注意力机制


- **CBAM**
   ：同时使用通道注意力和空间注意力
- **ECA-Net**
   ：用1D卷积替代全连接层，更高效的通道注意力
- **Coordinate Attention**
   ：沿水平和垂直方向分别进行池化，保留位置信息


| 模型 | 核心模块 | 参数量 | FLOPs | ImageNet Top-1 |
| --- | --- | --- | --- | --- |
| MobileNet v1 | 深度可分离卷积 | 4.2M | 569M | 70.6% |
| MobileNet v2 | Inverted Residual + Linear Bottleneck | 3.4M | 300M | 72.0% |
| MobileNet v3 | NAS + SE + h-swish | 5.4M | 219M | 75.2% |
| ShuffleNet v2 | Channel Shuffle + Split | 2.3M | 146M | 69.4% |
| GhostNet | Ghost Module | 5.2M | 141M | 73.9% |
| EfficientNet-B0 | MBConv + SE | 5.3M | 390M | 77.1% |


### 8.1 ViT的优缺点


**优势：**


- 全局感受野：每个token可以关注所有其他token
- 可扩展性强：模型越大、数据越多，性能持续提升
- 统一架构：图像、文本、多模态可使用同一框架


**劣势：**


- 缺少归纳偏置：不具有局部性和平移不变性
- 计算复杂度：自注意力为O(n²)，高分辨率图像代价大
- 需要大量数据：小数据集上性能不如CNN


### 8.2 融合架构


| 模型 | 策略 | 特点 |
| --- | --- | --- |
| Swin Transformer | 窗口注意力+移位 | 层次化结构，线性复杂度 |
| ConvNeXt | 用CNN模仿Transformer设计 | 纯CNN达到ViT级别性能 |
| CoAtNet | 前半CNN + 后半Transformer | 兼具两者优势 |
| EfficientFormer | 混合卷积和注意力 | 移动端友好 |
| MaxViT | 块注意力+网格注意力 | 多轴注意力机制 |


### 9.1 不同场景的推荐模型


| 场景 | 推荐模型 | 理由 |
| --- | --- | --- |
| 移动端/嵌入式 | MobileNet v3, EfficientNet-B0/B1 | 计算量小，精度合理 |
| 服务端通用分类 | ResNet-50, EfficientNet-B3/B4 | 性能与效率平衡 |
| 高精度要求 | EfficientNet-B7, ViT-L, ConvNeXt-XL | 追求最高准确率 |
| 小数据集 | ResNet-34/50 + 迁移学习 | 归纳偏置帮助泛化 |
| 实时推理 | MobileNet v2, ShuffleNet v2 | 低延迟 |


### 9.2 训练技巧


- **学习率策略**
   ：Warmup + Cosine Annealing 或 Step Decay
- **标签平滑（Label Smoothing）**
   ：将one-hot标签改为(1-ε)·y + ε/K
- **Mixup / CutMix**
   ：数据增强策略，混合样本和标签
- **EMA（指数移动平均）**
   ：维护模型参数的滑动平均
- **随机深度（Stochastic Depth）**
   ：训练时随机跳过部分层


<!-- Converted from: 01_CNN架构演进.html -->
