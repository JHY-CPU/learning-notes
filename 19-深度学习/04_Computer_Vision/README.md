# 04_Computer_Vision

计算机视觉：CNN架构、目标检测、分割、生成模型、3D视觉

共 70 篇笔记

| 编号 | 笔记 |
|------|------|
| 01 | [卷积运算的数学定义与互相关区别](01_卷积运算的数学定义与互相关区别.md) |
| 02 | [卷积核参数计算与感受野推导](02_卷积核参数计算与感受野推导.md) |
| 03 | [MaxPooling 与 AveragePooling 的差异及梯度流动](03_MaxPooling 与 AveragePooling 的差异及梯度流动.md) |
| 04 | [步长 (Stride) 与填充 (Padding) 对输出尺寸的影响](04_步长 (Stride) 与填充 (Padding) 对输出尺寸的影响.md) |
| 05 | [转置卷积 (Transposed Conv) 与棋盘效应](05_转置卷积 (Transposed Conv) 与棋盘效应.md) |
| 06 | [空洞卷积 (Dilated Conv) 扩大感受野](06_空洞卷积 (Dilated Conv) 扩大感受野.md) |
| 07 | [深度可分离卷积 (Depthwise Separable Conv)](07_深度可分离卷积 (Depthwise Separable Conv).md) |
| 08 | [分组卷积 (Group Conv) 与 ShuffleNet](08_分组卷积 (Group Conv) 与 ShuffleNet.md) |
| 09 | [LeNet-5：现代 CNN 的雏形](09_LeNet-5：现代 CNN 的雏形.md) |
| 10 | [AlexNet：ReLU、Dropout 与 GPU 训练](10_AlexNet：ReLU、Dropout 与 GPU 训练.md) |
| 11 | [VGGNet：小卷积核堆叠的设计哲学](11_VGGNet：小卷积核堆叠的设计哲学.md) |
| 12 | [GoogLeNet (Inception)：多尺度特征提取](12_GoogLeNet (Inception)：多尺度特征提取.md) |
| 13 | [ResNet：残差学习与退化问题解决](13_ResNet：残差学习与退化问题解决.md) |
| 14 | [ResNeXt：基数 (Cardinality) 的概念](14_ResNeXt：基数 (Cardinality) 的概念.md) |
| 15 | [DenseNet：密集连接与特征传播](15_DenseNet：密集连接与特征传播.md) |
| 16 | [MobileNet V1-V2-V3：移动端轻量化设计](16_MobileNet V1-V2-V3：移动端轻量化设计.md) |
| 17 | [EfficientNet：复合缩放系数 (Compound Scaling)](17_EfficientNet：复合缩放系数 (Compound Scaling).md) |
| 18 | [Vision Transformer (ViT) 架构详解](18_Vision Transformer (ViT) 架构详解.md) |
| 19 | [Swin Transformer：层级式设计与移位窗口](19_Swin Transformer：层级式设计与移位窗口.md) |
| 20 | [R-CNN：区域选择与特征提取的两阶段逻辑](20_R-CNN：区域选择与特征提取的两阶段逻辑.md) |
| 21 | [Fast R-CNN 与 ROI Pooling 实现](21_Fast R-CNN 与 ROI Pooling 实现.md) |
| 22 | [Faster R-CNN 与 RPN (Region Proposal Network)](22_Faster R-CNN 与 RPN (Region Proposal Network).md) |
| 23 | [Mask R-CNN：像素级实例分割](23_Mask R-CNN：像素级实例分割.md) |
| 24 | [YOLO 系列：单阶段检测的网格划分思想](24_YOLO 系列：单阶段检测的网格划分思想.md) |
| 25 | [YOLO 损失函数演变：GIOU, DIOU, CIOU](25_YOLO 损失函数演变：GIOU, DIOU, CIOU.md) |
| 26 | [SSD：多尺度特征图的目标检测](26_SSD：多尺度特征图的目标检测.md) |
| 27 | [Feature Pyramid Network (FPN) 特征金字塔](27_Feature Pyramid Network (FPN) 特征金字塔.md) |
| 28 | [U-Net：编码器-解码器结构与跳跃连接](28_U-Net：编码器-解码器结构与跳跃连接.md) |
| 29 | [DeepLab 系列：ASPP 与空洞空间金字塔池化](29_DeepLab 系列：ASPP 与空洞空间金字塔池化.md) |
| 30 | [PANet 与 FPN 的双向特征融合](30_PANet 与 FPN 的双向特征融合.md) |
| 31 | [图像分类中的标签平滑与 Mixup 增强](31_图像分类中的标签平滑与 Mixup 增强.md) |
| 32 | [CutMix 数据增强策略的实现细节](32_CutMix 数据增强策略的实现细节.md) |
| 33 | [风格迁移 (Style Transfer) 原理与 Gram 矩阵](33_风格迁移 (Style Transfer) 原理与 Gram 矩阵.md) |
| 34 | [超分辨率重建 (Super-Resolution) 基础](34_超分辨率重建 (Super-Resolution) 基础.md) |
| 35 | [人脸对齐与关键点检测技术](35_人脸对齐与关键点检测技术.md) |
| 36 | [光流估计 (Optical Flow) 与运动分析](36_光流估计 (Optical Flow) 与运动分析.md) |
| 37 | [3D CNN 与视频动作识别 (C3D, I3D)](37_3D CNN 与视频动作识别 (C3D, I3D).md) |
| 38 | [TimeSformer：视频中的时空注意力](38_TimeSformer：视频中的时空注意力.md) |
| 39 | [点云处理：PointNet 与 PointNet++](39_点云处理：PointNet 与 PointNet++.md) |
| 40 | [体素化 (Voxelization) 与 3D 卷积](40_体素化 (Voxelization) 与 3D 卷积.md) |
| 41 | [神经辐射场 (NeRF) 基础原理](41_神经辐射场 (NeRF) 基础原理.md) |
| 42 | [3D Gaussian Splatting 渲染技术](42_3D Gaussian Splatting 渲染技术.md) |
| 43 | [图像检索与哈希学习 (Hash Learning)](43_图像检索与哈希学习 (Hash Learning).md) |
| 44 | [零样本学习 (Zero-Shot Learning) 在 CV 中的应用](44_零样本学习 (Zero-Shot Learning) 在 CV 中的应用.md) |
| 45 | [自监督视觉预训练：MAE (Masked Autoencoders)](45_自监督视觉预训练：MAE (Masked Autoencoders).md) |
| 46 | [DINOv2：无需标签的视觉特征提取](46_DINOv2：无需标签的视觉特征提取.md) |
| 47 | [CLIP 模型：图文对齐与多模态理解](47_CLIP 模型：图文对齐与多模态理解.md) |
| 48 | [Stable Diffusion 在图像编辑中的应用](48_Stable Diffusion 在图像编辑中的应用.md) |
| 49 | [ControlNet：精确控制生成图像结构](49_ControlNet：精确控制生成图像结构.md) |
| 50 | [Segment Anything Model (SAM) 架构分析](50_Segment Anything Model (SAM) 架构分析.md) |
| 51 | [视觉问答 (VQA) 系统设计](51_视觉问答 (VQA) 系统设计.md) |
| 52 | [图像描述 (Image Captioning) 生成技术](52_图像描述 (Image Captioning) 生成技术.md) |
| 53 | [OCR 光学字符识别与 CRNN 模型](53_OCR 光学字符识别与 CRNN 模型.md) |
| 54 | [车牌识别与场景文本检测 (DBNet)](54_车牌识别与场景文本检测 (DBNet).md) |
| 55 | [医学图像分割中的 Dice Loss](55_医学图像分割中的 Dice Loss.md) |
| 56 | [遥感图像中的旋转目标检测](56_遥感图像中的旋转目标检测.md) |
| 57 | [工业缺陷检测中的小样本问题](57_工业缺陷检测中的小样本问题.md) |
| 58 | [自动驾驶中的多任务学习网络](58_自动驾驶中的多任务学习网络.md) |
| 59 | [BEV (Bird's Eye View) 视角转换技术](59_BEV (Bird's Eye View) 视角转换技术.md) |
| 60 | [视觉 SLAM 与深度学习结合](60_视觉 SLAM 与深度学习结合.md) |
| 61 | [图像插值算法：双线性与双三次插值](61_图像插值算法：双线性与双三次插值.md) |
| 62 | [颜色空间转换：RGB, HSV, LAB 的差异](62_颜色空间转换：RGB, HSV, LAB 的差异.md) |
| 63 | [边缘检测算子：Sobel, Canny 原理](63_边缘检测算子：Sobel, Canny 原理.md) |
| 64 | [Hough 变换与直线及圆检测](64_Hough 变换与直线及圆检测.md) |
| 65 | [SIFT 特征提取与匹配](65_SIFT 特征提取与匹配.md) |
| 66 | [ORB 特征点检测与描述子](66_ORB 特征点检测与描述子.md) |
| 67 | [相机标定与内参及外参矩阵](67_相机标定与内参及外参矩阵.md) |
| 68 | [立体视觉与深度图生成](68_立体视觉与深度图生成.md) |
| 69 | [全景拼接与图像配准](69_全景拼接与图像配准.md) |
| 70 | [视频目标跟踪：Siamese 网络](70_视频目标跟踪：Siamese 网络.md) |
