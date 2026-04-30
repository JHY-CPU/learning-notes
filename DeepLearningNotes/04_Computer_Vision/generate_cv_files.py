import os

titles = [
    "01_卷积运算的数学定义与互相关区别",
    "02_卷积核参数计算与感受野推导",
    "03_MaxPooling 与 AveragePooling 的差异及梯度流动",
    "04_步长 (Stride) 与填充 (Padding) 对输出尺寸的影响",
    "05_转置卷积 (Transposed Conv) 与棋盘效应",
    "06_空洞卷积 (Dilated Conv) 扩大感受野",
    "07_深度可分离卷积 (Depthwise Separable Conv)",
    "08_分组卷积 (Group Conv) 与 ShuffleNet",
    "09_LeNet-5：现代 CNN 的雏形",
    "10_AlexNet：ReLU、Dropout 与 GPU 训练",
    "11_VGGNet：小卷积核堆叠的设计哲学",
    "12_GoogLeNet (Inception)：多尺度特征提取",
    "13_ResNet：残差学习与退化问题解决",
    "14_ResNeXt：基数 (Cardinality) 的概念",
    "15_DenseNet：密集连接与特征传播",
    "16_MobileNet V1-V2-V3：移动端轻量化设计",
    "17_EfficientNet：复合缩放系数 (Compound Scaling)",
    "18_Vision Transformer (ViT) 架构详解",
    "19_Swin Transformer：层级式设计与移位窗口",
    "20_R-CNN：区域选择与特征提取的两阶段逻辑",
    "21_Fast R-CNN 与 ROI Pooling 实现",
    "22_Faster R-CNN 与 RPN (Region Proposal Network)",
    "23_Mask R-CNN：像素级实例分割",
    "24_YOLO 系列：单阶段检测的网格划分思想",
    "25_YOLO 损失函数演变：GIOU, DIOU, CIOU",
    "26_SSD：多尺度特征图的目标检测",
    "27_Feature Pyramid Network (FPN) 特征金字塔",
    "28_U-Net：编码器-解码器结构与跳跃连接",
    "29_DeepLab 系列：ASPP 与空洞空间金字塔池化",
    "30_PANet 与 FPN 的双向特征融合",
    "31_图像分类中的标签平滑与 Mixup 增强",
    "32_CutMix 数据增强策略的实现细节",
    "33_风格迁移 (Style Transfer) 原理与 Gram 矩阵",
    "34_超分辨率重建 (Super-Resolution) 基础",
    "35_人脸对齐与关键点检测技术",
    "36_光流估计 (Optical Flow) 与运动分析",
    "37_3D CNN 与视频动作识别 (C3D, I3D)",
    "38_TimeSformer：视频中的时空注意力",
    "39_点云处理：PointNet 与 PointNet++",
    "40_体素化 (Voxelization) 与 3D 卷积"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
