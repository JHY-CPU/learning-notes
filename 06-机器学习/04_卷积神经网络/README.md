# 卷积神经网络

## 一、卷积操作

### 1.1 2D卷积

$$O(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)$$

输出尺寸：$O = (W - K + 2P) / S + 1$

参数共享大幅减少参数量：传统全连接需要 $W \times H \times C_{in} \times C_{out}$ 个参数，卷积仅需 $K^2 \times C_{in} \times C_{out}$。

### 1.2 多通道卷积

输入 $C_{in}$ 通道，使用 $C_{out}$ 个卷积核，输出 $C_{out}$ 通道。每个输出通道的卷积核是 $C_{in} \times K \times K$。

---

## 二、经典架构

### 2.1 LeNet-5（1998）

两个卷积层 + 三个全连接层，用于手写数字识别。

### 2.2 AlexNet（2012）

ReLU激活、Dropout、数据增强、GPU训练，ImageNet错误率从26%降至16%。

### 2.3 VGGNet（2014）

使用3×3小卷积核堆叠，证明了深度的重要性。

### 2.4 GoogLeNet/Inception（2014）

Inception模块：并行使用1×1、3×3、5×5卷积和池化，多尺度特征提取。

### 2.5 ResNet（2015）

残差连接 $y = F(x) + x$：
- 解决深层网络退化问题
- 训练152层甚至1000+层网络
- 梯度可以通过跳跃连接直接传播

### 2.6 EfficientNet（2019）

复合缩放：同时缩放分辨率、宽度和深度。

---

## 三、目标检测

- **R-CNN系列**：两阶段方法，先生成候选框再分类
- **YOLO系列**：单阶段方法，一次前向传播完成检测
- **SSD**：多尺度检测

---

## 四、Vision Transformer

将图像分割为patch，用Transformer编码：
- ViT：直接应用标准Transformer
- Swin Transformer：窗口注意力，层次化结构
- DeiT：数据高效的ViT训练
