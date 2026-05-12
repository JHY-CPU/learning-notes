# 03_神经网络核心

> 深度神经网络的核心组件与技术。从感知机到现代深度网络架构，系统覆盖前向/反向传播、激活函数、损失函数、优化器、正则化、归一化技术，以及高级训练技巧和前沿研究方向。

---

## 基础知识

- **前置知识**：01_数学基础; 02_机器学习基础; PyTorch 基础用法
- **关联目录**：04_计算机视觉（CV 中的网络设计）; 05_NLP与序列模型（NLP 中的网络设计）; 08_工程与部署（训练工程化）
- **笔记数量**：共 60 篇

---

## 内容结构

#### 网络基础与传播算法

感知机、MLP、万能近似定理、前向/反向传播、梯度推导

| 编号 | 笔记 |
|------|------|
| 00 | [感知机模型与逻辑门实现](0_感知机模型与逻辑门实现.md) |
| 01 | [多层感知机 (MLP) 的万能近似定理](1_多层感知机 (MLP) 的万能近似定理.md) |
| 02 | [前向传播的计算图表示](2_前向传播的计算图表示.md) |
| 03 | [反向传播算法：链式法则的递归应用](3_反向传播算法：链式法则的递归应用.md) |
| 04 | [全连接层梯度推导：权重与偏置](4_全连接层梯度推导：权重与偏置.md) |

#### 激活函数与损失函数

Sigmoid/Tanh/ReLU/GELU、Softmax、MSE/Cross-Entropy/Huber/Triplet Loss

| 编号 | 笔记 |
|------|------|
| 05 | [Sigmoid 激活函数与梯度消失问题](5_Sigmoid 激活函数与梯度消失问题.md) |
| 06 | [Tanh 激活函数的零中心特性](6_Tanh 激活函数的零中心特性.md) |
| 07 | [ReLU 及其变体 (Leaky ReLU, PReLU)](7_ReLU 及其变体 (Leaky ReLU, PReLU).md) |
| 08 | [GELU 与 Swish：现代 Transformer 的首选](8_GELU 与 Swish：现代 Transformer 的首选.md) |
| 09 | [Softmax 函数推导与数值稳定性 (Log-Sum-Exp)](9_Softmax 函数推导与数值稳定性 (Log-Sum-Exp).md) |
| 10 | [均方误差 (MSE) 损失函数的梯度特性](10_均方误差 (MSE) 损失函数的梯度特性.md) |
| 11 | [交叉熵损失 (Cross-Entropy) 的导数简化形式](11_交叉熵损失 (Cross-Entropy) 的导数简化形式.md) |
| 12 | [Huber Loss 对异常值的鲁棒性](12_Huber Loss 对异常值的鲁棒性.md) |
| 13 | [Contrastive Loss 与相似度学习](13_Contrastive Loss 与相似度学习.md) |
| 14 | [Triplet Loss 与锚点选择策略](14_Triplet Loss 与锚点选择策略.md) |

#### 优化器

SGD、Momentum、NAG、AdaGrad、RMSProp、Adam、AdamW、LAMB

| 编号 | 笔记 |
|------|------|
| 15 | [SGD 优化器：随机性与逃逸局部极小值](15_SGD 优化器：随机性与逃逸局部极小值.md) |
| 16 | [Momentum 动量项的物理类比与公式](16_Momentum 动量项的物理类比与公式.md) |
| 17 | [Nesterov Accelerated Gradient (NAG) 的前瞻性](17_Nesterov Accelerated Gradient (NAG) 的前瞻性.md) |
| 18 | [AdaGrad：自适应学习率与稀疏特征](18_AdaGrad：自适应学习率与稀疏特征.md) |
| 19 | [RMSProp：解决 AdaGrad 学习率单调递减](19_RMSProp：解决 AdaGrad 学习率单调递减.md) |
| 20 | [Adam 优化器：一阶与二阶矩估计](20_Adam 优化器：一阶与二阶矩估计.md) |
| 21 | [AdamW：解耦权重衰减 (Weight Decay)](21_AdamW：解耦权重衰减 (Weight Decay).md) |
| 22 | [LAMB 与 LARS：大 Batch 训练优化器](22_LAMB 与 LARS：大 Batch 训练优化器.md) |

#### 参数初始化

Xavier、He、正交初始化

| 编号 | 笔记 |
|------|------|
| 23 | [Xavier 初始化与方差保持推导](23_Xavier 初始化与方差保持推导.md) |
| 24 | [He 初始化与 ReLU 的适配性](24_He 初始化与 ReLU 的适配性.md) |
| 25 | [正交初始化 (Orthogonal Initialization) 在 RNN 中的应用](25_正交初始化 (Orthogonal Initialization) 在 RNN 中的应用.md) |

#### 正则化技术

L1/L2、Early Stopping、Dropout、数据增强、Label Smoothing

| 编号 | 笔记 |
|------|------|
| 26 | [L2 正则化与权重衰减的等价性证明](26_L2 正则化与权重衰减的等价性证明.md) |
| 27 | [L1 正则化与稀疏解的几何解释](27_L1 正则化与稀疏解的几何解释.md) |
| 28 | [Early Stopping 作为隐式正则化](28_Early Stopping 作为隐式正则化.md) |
| 29 | [Dropout：集成学习的视角与 Inference 缩放](29_Dropout：集成学习的视角与 Inference 缩放.md) |
| 30 | [Data Augmentation 数据增强的正则化效果](30_Data Augmentation 数据增强的正则化效果.md) |
| 31 | [Label Smoothing 标签平滑的原理与实现](31_Label Smoothing 标签平滑的原理与实现.md) |

#### 归一化与跳跃连接

BatchNorm、LayerNorm、InstanceNorm、GroupNorm、ResNet、DenseNet

| 编号 | 笔记 |
|------|------|
| 32 | [Batch Normalization：前向与反向传播推导](32_Batch Normalization：前向与反向传播推导.md) |
| 33 | [Layer Normalization 与 BatchNorm 的对比](33_Layer Normalization 与 BatchNorm 的对比.md) |
| 34 | [Instance Normalization 与 Style Transfer](34_Instance Normalization 与 Style Transfer.md) |
| 35 | [Group Normalization：小 Batch 下的替代方案](35_Group Normalization：小 Batch 下的替代方案.md) |
| 36 | [Residual Connection 残差连接的梯度流动分析](36_Residual Connection 残差连接的梯度流动分析.md) |
| 37 | [DenseNet 密集连接与特征复用](37_DenseNet 密集连接与特征复用.md) |

#### 高级主题

NTK、Loss Landscape、梯度裁剪、学习率调度、混合精度、Autograd

| 编号 | 笔记 |
|------|------|
| 38 | [Neural Tangent Kernel (NTK) 理论基础](38_Neural Tangent Kernel (NTK) 理论基础.md) |
| 39 | [损失曲面 (Loss Landscape) 可视化与分析](39_损失曲面 (Loss Landscape) 可视化与分析.md) |
| 40 | [梯度裁剪 (Gradient Clipping) 防止爆炸](40_梯度裁剪 (Gradient Clipping) 防止爆炸.md) |
| 41 | [学习率调度器：StepLR 与 MultiStepLR](41_学习率调度器：StepLR 与 MultiStepLR.md) |
| 42 | [Cosine Annealing 余弦退火策略](42_Cosine Annealing 余弦退火策略.md) |
| 43 | [One Cycle Learning Rate Policy](43_One Cycle Learning Rate Policy.md) |
| 44 | [Warmup 预热阶段的必要性分析](44_Warmup 预热阶段的必要性分析.md) |
| 45 | [混合精度训练 (Mixed Precision) 原理](45_混合精度训练 (Mixed Precision) 原理.md) |
| 46 | [损失缩放 (Loss Scaling) 技术细节](46_损失缩放 (Loss Scaling) 技术细节.md) |
| 47 | [动态计算图与静态计算图的对比](47_动态计算图与静态计算图的对比.md) |
| 48 | [PyTorch Autograd 引擎底层逻辑](48_PyTorch Autograd 引擎底层逻辑.md) |
| 49 | [TensorFlow GradientTape 机制](49_TensorFlow GradientTape 机制.md) |

#### 前沿方向

可逆网络、Deep Equilibrium、Neural ODE、元学习、少样本学习、自监督学习、知识蒸馏

| 编号 | 笔记 |
|------|------|
| 50 | [神经网络的可逆性 (Invertible Networks)](50_神经网络的可逆性 (Invertible Networks).md) |
| 51 | [深度平衡模型 (Deep Equilibrium Models)](51_深度平衡模型 (Deep Equilibrium Models).md) |
| 52 | [神经微分方程 (Neural ODEs) 基础](52_神经微分方程 (Neural ODEs) 基础.md) |
| 53 | [元学习 (Meta-Learning) 与 MAML 算法](53_元学习 (Meta-Learning) 与 MAML 算法.md) |
| 54 | [少样本学习 (Few-Shot Learning) 范式](54_少样本学习 (Few-Shot Learning) 范式.md) |
| 55 | [自监督学习 (Self-Supervised Learning) 概述](55_自监督学习 (Self-Supervised Learning) 概述.md) |
| 56 | [对比学习 (Contrastive Learning) 框架：SimCLR](56_对比学习 (Contrastive Learning) 框架：SimCLR.md) |
| 57 | [知识蒸馏 (Knowledge Distillation) 损失设计](57_知识蒸馏 (Knowledge Distillation) 损失设计.md) |
| 58 | [模型集成 (Ensembling) 的策略与收益](58_模型集成 (Ensembling) 的策略与收益.md) |
| 59 | [神经架构搜索 (NAS) 基础方法](59_神经架构搜索 (NAS) 基础方法.md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
