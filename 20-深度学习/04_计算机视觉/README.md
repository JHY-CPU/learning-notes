# 04_计算机视觉

> 从经典卷积网络到现代视觉Transformer，从基础图像处理到3D视觉与多模态理解。本目录覆盖了计算机视觉的核心技术栈：CNN架构、目标检测、语义分割、图像生成、3D重建、视觉-语言模型。

---

## 基础知识

- **前置知识**：03_神经网络核心; 01_数学基础（线性代数、傅里叶变换）
- **关联目录**：05_NLP与序列模型（多模态中的文本编码）; 06_生成式AI（图像生成）
- **笔记数量**：共 70 篇

---

## 内容结构

#### 卷积基础与变体

卷积运算、池化、步长填充、转置卷积、空洞卷积、深度可分离卷积、分组卷积

| 编号 | 笔记 |
|------|------|
| 00 | [卷积运算的数学定义与互相关区别](0_卷积运算的数学定义与互相关区别.md) |
| 01 | [卷积核参数计算与感受野推导](1_卷积核参数计算与感受野推导.md) |
| 02 | [MaxPooling 与 AveragePooling 的差异及梯度流动](2_MaxPooling 与 AveragePooling 的差异及梯度流动.md) |
| 03 | [步长 (Stride) 与填充 (Padding) 对输出尺寸的影响](3_步长 (Stride) 与填充 (Padding) 对输出尺寸的影响.md) |
| 04 | [转置卷积 (Transposed Conv) 与棋盘效应](4_转置卷积 (Transposed Conv) 与棋盘效应.md) |
| 05 | [空洞卷积 (Dilated Conv) 扩大感受野](5_空洞卷积 (Dilated Conv) 扩大感受野.md) |
| 06 | [深度可分离卷积 (Depthwise Separable Conv)](6_深度可分离卷积 (Depthwise Separable Conv).md) |
| 07 | [分组卷积 (Group Conv) 与 ShuffleNet](7_分组卷积 (Group Conv) 与 ShuffleNet.md) |

#### 经典与现代架构

LeNet、AlexNet、VGG、GoogLeNet、ResNet、DenseNet、MobileNet、EfficientNet、ViT、Swin

| 编号 | 笔记 |
|------|------|
| 08 | [LeNet-5：现代 CNN 的雏形](8_LeNet-5：现代 CNN 的雏形.md) |
| 09 | [AlexNet：ReLU、Dropout 与 GPU 训练](9_AlexNet：ReLU、Dropout 与 GPU 训练.md) |
| 10 | [VGGNet：小卷积核堆叠的设计哲学](10_VGGNet：小卷积核堆叠的设计哲学.md) |
| 11 | [GoogLeNet (Inception)：多尺度特征提取](11_GoogLeNet (Inception)：多尺度特征提取.md) |
| 12 | [ResNet：残差学习与退化问题解决](12_ResNet：残差学习与退化问题解决.md) |
| 13 | [ResNeXt：基数 (Cardinality) 的概念](13_ResNeXt：基数 (Cardinality) 的概念.md) |
| 14 | [DenseNet：密集连接与特征传播](14_DenseNet：密集连接与特征传播.md) |
| 15 | [MobileNet V1-V2-V3：移动端轻量化设计](15_MobileNet V1-V2-V3：移动端轻量化设计.md) |
| 16 | [EfficientNet：复合缩放系数 (Compound Scaling)](16_EfficientNet：复合缩放系数 (Compound Scaling).md) |
| 17 | [Vision Transformer (ViT) 架构详解](17_Vision Transformer (ViT) 架构详解.md) |
| 18 | [Swin Transformer：层级式设计与移位窗口](18_Swin Transformer：层级式设计与移位窗口.md) |

#### 目标检测与分割

R-CNN系列、YOLO、SSD、FPN、U-Net、DeepLab、PANet

| 编号 | 笔记 |
|------|------|
| 19 | [R-CNN：区域选择与特征提取的两阶段逻辑](19_R-CNN：区域选择与特征提取的两阶段逻辑.md) |
| 20 | [Fast R-CNN 与 ROI Pooling 实现](20_Fast R-CNN 与 ROI Pooling 实现.md) |
| 21 | [Faster R-CNN 与 RPN (Region Proposal Network)](21_Faster R-CNN 与 RPN (Region Proposal Network).md) |
| 22 | [Mask R-CNN：像素级实例分割](22_Mask R-CNN：像素级实例分割.md) |
| 23 | [YOLO 系列：单阶段检测的网格划分思想](23_YOLO 系列：单阶段检测的网格划分思想.md) |
| 24 | [YOLO 损失函数演变：GIOU, DIOU, CIOU](24_YOLO 损失函数演变：GIOU, DIOU, CIOU.md) |
| 25 | [SSD：多尺度特征图的目标检测](25_SSD：多尺度特征图的目标检测.md) |
| 26 | [Feature Pyramid Network (FPN) 特征金字塔](26_Feature Pyramid Network (FPN) 特征金字塔.md) |
| 27 | [U-Net：编码器-解码器结构与跳跃连接](27_U-Net：编码器-解码器结构与跳跃连接.md) |
| 28 | [DeepLab 系列：ASPP 与空洞空间金字塔池化](28_DeepLab 系列：ASPP 与空洞空间金字塔池化.md) |
| 29 | [PANet 与 FPN 的双向特征融合](29_PANet 与 FPN 的双向特征融合.md) |

#### 图像处理与3D视觉

数据增强、风格迁移、超分辨率、人脸检测、光流、3D CNN、点云、NeRF

| 编号 | 笔记 |
|------|------|
| 30 | [图像分类中的标签平滑与 Mixup 增强](30_图像分类中的标签平滑与 Mixup 增强.md) |
| 31 | [CutMix 数据增强策略的实现细节](31_CutMix 数据增强策略的实现细节.md) |
| 32 | [风格迁移 (Style Transfer) 原理与 Gram 矩阵](32_风格迁移 (Style Transfer) 原理与 Gram 矩阵.md) |
| 33 | [超分辨率重建 (Super-Resolution) 基础](33_超分辨率重建 (Super-Resolution) 基础.md) |
| 34 | [人脸对齐与关键点检测技术](34_人脸对齐与关键点检测技术.md) |
| 35 | [光流估计 (Optical Flow) 与运动分析](35_光流估计 (Optical Flow) 与运动分析.md) |
| 36 | [3D CNN 与视频动作识别 (C3D, I3D)](36_3D CNN 与视频动作识别 (C3D, I3D).md) |
| 37 | [TimeSformer：视频中的时空注意力](37_TimeSformer：视频中的时空注意力.md) |
| 38 | [点云处理：PointNet 与 PointNet++](38_点云处理：PointNet 与 PointNet++.md) |
| 39 | [体素化 (Voxelization) 与 3D 卷积](39_体素化 (Voxelization) 与 3D 卷积.md) |

#### 多模态与自监督

3D Gaussian Splatting、图像检索、零样本、MAE、DINOv2、CLIP、Stable Diffusion、SAM

| 编号 | 笔记 |
|------|------|
| 40 | [神经辐射场 (NeRF) 基础原理](40_神经辐射场 (NeRF) 基础原理.md) |
| 41 | [3D Gaussian Splatting 渲染技术](41_3D Gaussian Splatting 渲染技术.md) |
| 42 | [图像检索与哈希学习 (Hash Learning)](42_图像检索与哈希学习 (Hash Learning).md) |
| 43 | [零样本学习 (Zero-Shot Learning) 在 CV 中的应用](43_零样本学习 (Zero-Shot Learning) 在 CV 中的应用.md) |
| 44 | [自监督视觉预训练：MAE (Masked Autoencoders)](44_自监督视觉预训练：MAE (Masked Autoencoders).md) |
| 45 | [DINOv2：无需标签的视觉特征提取](45_DINOv2：无需标签的视觉特征提取.md) |
| 46 | [CLIP 模型：图文对齐与多模态理解](46_CLIP 模型：图文对齐与多模态理解.md) |
| 47 | [Stable Diffusion 在图像编辑中的应用](47_Stable Diffusion 在图像编辑中的应用.md) |
| 48 | [ControlNet：精确控制生成图像结构](48_ControlNet：精确控制生成图像结构.md) |
| 49 | [Segment Anything Model (SAM) 架构分析](49_Segment Anything Model (SAM) 架构分析.md) |

#### 应用系统

VQA、图像描述、OCR、场景文本检测、医学图像分割、自动驾驶、BEV、SLAM

| 编号 | 笔记 |
|------|------|
| 50 | [视觉问答 (VQA) 系统设计](50_视觉问答 (VQA) 系统设计.md) |
| 51 | [图像描述 (Image Captioning) 生成技术](51_图像描述 (Image Captioning) 生成技术.md) |
| 52 | [OCR 光学字符识别与 CRNN 模型](52_OCR 光学字符识别与 CRNN 模型.md) |
| 53 | [车牌识别与场景文本检测 (DBNet)](53_车牌识别与场景文本检测 (DBNet).md) |
| 54 | [医学图像分割中的 Dice Loss](54_医学图像分割中的 Dice Loss.md) |
| 55 | [遥感图像中的旋转目标检测](55_遥感图像中的旋转目标检测.md) |
| 56 | [工业缺陷检测中的小样本问题](56_工业缺陷检测中的小样本问题.md) |
| 57 | [自动驾驶中的多任务学习网络](57_自动驾驶中的多任务学习网络.md) |
| 58 | [BEV (Bird's Eye View) 视角转换技术](58_BEV (Bird's Eye View) 视角转换技术.md) |
| 59 | [视觉 SLAM 与深度学习结合](59_视觉 SLAM 与深度学习结合.md) |

#### 经典图像处理

插值算法、颜色空间、边缘检测、Hough变换、SIFT、ORB、相机标定、立体视觉

| 编号 | 笔记 |
|------|------|
| 60 | [图像插值算法：双线性与双三次插值](60_图像插值算法：双线性与双三次插值.md) |
| 61 | [颜色空间转换：RGB, HSV, LAB 的差异](61_颜色空间转换：RGB, HSV, LAB 的差异.md) |
| 62 | [边缘检测算子：Sobel, Canny 原理](62_边缘检测算子：Sobel, Canny 原理.md) |
| 63 | [Hough 变换与直线及圆检测](63_Hough 变换与直线及圆检测.md) |
| 64 | [SIFT 特征提取与匹配](64_SIFT 特征提取与匹配.md) |
| 65 | [ORB 特征点检测与描述子](65_ORB 特征点检测与描述子.md) |
| 66 | [相机标定与内参及外参矩阵](66_相机标定与内参及外参矩阵.md) |
| 67 | [立体视觉与深度图生成](67_立体视觉与深度图生成.md) |
| 68 | [全景拼接与图像配准](68_全景拼接与图像配准.md) |
| 69 | [视频目标跟踪：Siamese 网络](69_视频目标跟踪：Siamese 网络.md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
