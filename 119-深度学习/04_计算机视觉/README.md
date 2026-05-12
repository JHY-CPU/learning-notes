# 计算机视觉

## 一、卷积神经网络（CNN）

### 1.1 核心组件

- **卷积层**：通过卷积核提取局部特征，具有参数共享和局部连接的特点
- **池化层**：降低空间维度，增加平移不变性（Max Pooling、Average Pooling）
- **全连接层**：将特征图展平进行分类或回归

**输出尺寸计算：**
$$O = \frac{W - K + 2P}{S} + 1$$

其中 $W$ 为输入尺寸，$K$ 为卷积核大小，$P$ 为填充，$S$ 为步长。

### 1.2 经典架构演进

| 模型 | 年份 | 核心贡献 |
|------|------|----------|
| LeNet-5 | 1998 | 首个成功CNN，手写数字识别 |
| AlexNet | 2012 | ReLU、Dropout、GPU训练，ImageNet突破 |
| VGGNet | 2014 | 小卷积核(3x3)堆叠，更深网络 |
| GoogLeNet | 2014 | Inception模块，多尺度特征 |
| ResNet | 2015 | 残差连接，解决深层网络退化 |
| DenseNet | 2017 | 密集连接，特征复用 |
| EfficientNet | 2019 | 复合缩放策略 |

---

## 二、目标检测

### 2.1 两阶段方法

- **R-CNN**：Selective Search + CNN特征提取
- **Fast R-CNN**：共享卷积特征，ROI Pooling
- **Faster R-CNN**：RPN生成候选框，端到端训练

### 2.2 单阶段方法

- **YOLO**：将检测作为回归问题，一次前向传播
- **SSD**：多尺度特征图检测
- **RetinaNet**：Focal Loss解决类别不平衡

---

## 三、图像分割

- **语义分割**：FCN、U-Net、DeepLab（ASPP、空洞卷积）
- **实例分割**：Mask R-CNN
- **全景分割**：Panoptic FPN

---

## 四、Vision Transformer (ViT)

将图像切分为patch，通过Transformer编码：

1. 图像分为 $16 \times 16$ 的patch
2. 每个patch线性嵌入 + 位置编码
3. 标准Transformer Encoder
4. [CLS] token用于分类

代表模型：ViT、DeiT、Swin Transformer（窗口注意力）
