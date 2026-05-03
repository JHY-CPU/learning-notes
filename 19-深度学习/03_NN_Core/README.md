# 03_NN_Core

神经网络核心：前向/反向传播、激活函数、损失函数、优化器、正则化、归一化

共 60 篇笔记

| 编号 | 笔记 |
|------|------|
| 01 | [感知机模型与逻辑门实现](01_感知机模型与逻辑门实现.md) |
| 02 | [多层感知机 (MLP) 的万能近似定理](02_多层感知机 (MLP) 的万能近似定理.md) |
| 03 | [前向传播的计算图表示](03_前向传播的计算图表示.md) |
| 04 | [反向传播算法：链式法则的递归应用](04_反向传播算法：链式法则的递归应用.md) |
| 05 | [全连接层梯度推导：权重与偏置](05_全连接层梯度推导：权重与偏置.md) |
| 06 | [Sigmoid 激活函数与梯度消失问题](06_Sigmoid 激活函数与梯度消失问题.md) |
| 07 | [Tanh 激活函数的零中心特性](07_Tanh 激活函数的零中心特性.md) |
| 08 | [ReLU 及其变体 (Leaky ReLU, PReLU)](08_ReLU 及其变体 (Leaky ReLU, PReLU).md) |
| 09 | [GELU 与 Swish：现代 Transformer 的首选](09_GELU 与 Swish：现代 Transformer 的首选.md) |
| 10 | [Softmax 函数推导与数值稳定性 (Log-Sum-Exp)](10_Softmax 函数推导与数值稳定性 (Log-Sum-Exp).md) |
| 11 | [均方误差 (MSE) 损失函数的梯度特性](11_均方误差 (MSE) 损失函数的梯度特性.md) |
| 12 | [交叉熵损失 (Cross-Entropy) 的导数简化形式](12_交叉熵损失 (Cross-Entropy) 的导数简化形式.md) |
| 13 | [Huber Loss 对异常值的鲁棒性](13_Huber Loss 对异常值的鲁棒性.md) |
| 14 | [Contrastive Loss 与相似度学习](14_Contrastive Loss 与相似度学习.md) |
| 15 | [Triplet Loss 与锚点选择策略](15_Triplet Loss 与锚点选择策略.md) |
| 16 | [SGD 优化器：随机性与逃逸局部极小值](16_SGD 优化器：随机性与逃逸局部极小值.md) |
| 17 | [Momentum 动量项的物理类比与公式](17_Momentum 动量项的物理类比与公式.md) |
| 18 | [Nesterov Accelerated Gradient (NAG) 的前瞻性](18_Nesterov Accelerated Gradient (NAG) 的前瞻性.md) |
| 19 | [AdaGrad：自适应学习率与稀疏特征](19_AdaGrad：自适应学习率与稀疏特征.md) |
| 20 | [RMSProp：解决 AdaGrad 学习率单调递减](20_RMSProp：解决 AdaGrad 学习率单调递减.md) |
| 21 | [Adam 优化器：一阶与二阶矩估计](21_Adam 优化器：一阶与二阶矩估计.md) |
| 22 | [AdamW：解耦权重衰减 (Weight Decay)](22_AdamW：解耦权重衰减 (Weight Decay).md) |
| 23 | [LAMB 与 LARS：大 Batch 训练优化器](23_LAMB 与 LARS：大 Batch 训练优化器.md) |
| 24 | [Xavier 初始化与方差保持推导](24_Xavier 初始化与方差保持推导.md) |
| 25 | [He 初始化与 ReLU 的适配性](25_He 初始化与 ReLU 的适配性.md) |
| 26 | [正交初始化 (Orthogonal Initialization) 在 RNN 中的应用](26_正交初始化 (Orthogonal Initialization) 在 RNN 中的应用.md) |
| 27 | [L2 正则化与权重衰减的等价性证明](27_L2 正则化与权重衰减的等价性证明.md) |
| 28 | [L1 正则化与稀疏解的几何解释](28_L1 正则化与稀疏解的几何解释.md) |
| 29 | [Early Stopping 作为隐式正则化](29_Early Stopping 作为隐式正则化.md) |
| 30 | [Dropout：集成学习的视角与 Inference 缩放](30_Dropout：集成学习的视角与 Inference 缩放.md) |
| 31 | [Data Augmentation 数据增强的正则化效果](31_Data Augmentation 数据增强的正则化效果.md) |
| 32 | [Label Smoothing 标签平滑的原理与实现](32_Label Smoothing 标签平滑的原理与实现.md) |
| 33 | [Batch Normalization：前向与反向传播推导](33_Batch Normalization：前向与反向传播推导.md) |
| 34 | [Layer Normalization 与 BatchNorm 的对比](34_Layer Normalization 与 BatchNorm 的对比.md) |
| 35 | [Instance Normalization 与 Style Transfer](35_Instance Normalization 与 Style Transfer.md) |
| 36 | [Group Normalization：小 Batch 下的替代方案](36_Group Normalization：小 Batch 下的替代方案.md) |
| 37 | [Residual Connection 残差连接的梯度流动分析](37_Residual Connection 残差连接的梯度流动分析.md) |
| 38 | [DenseNet 密集连接与特征复用](38_DenseNet 密集连接与特征复用.md) |
| 39 | [Neural Tangent Kernel (NTK) 理论基础](39_Neural Tangent Kernel (NTK) 理论基础.md) |
| 40 | [损失曲面 (Loss Landscape) 可视化与分析](40_损失曲面 (Loss Landscape) 可视化与分析.md) |
| 41 | [梯度裁剪 (Gradient Clipping) 防止爆炸](41_梯度裁剪 (Gradient Clipping) 防止爆炸.md) |
| 42 | [学习率调度器：StepLR 与 MultiStepLR](42_学习率调度器：StepLR 与 MultiStepLR.md) |
| 43 | [Cosine Annealing 余弦退火策略](43_Cosine Annealing 余弦退火策略.md) |
| 44 | [One Cycle Learning Rate Policy](44_One Cycle Learning Rate Policy.md) |
| 45 | [Warmup 预热阶段的必要性分析](45_Warmup 预热阶段的必要性分析.md) |
| 46 | [混合精度训练 (Mixed Precision) 原理](46_混合精度训练 (Mixed Precision) 原理.md) |
| 47 | [损失缩放 (Loss Scaling) 技术细节](47_损失缩放 (Loss Scaling) 技术细节.md) |
| 48 | [动态计算图与静态计算图的对比](48_动态计算图与静态计算图的对比.md) |
| 49 | [PyTorch Autograd 引擎底层逻辑](49_PyTorch Autograd 引擎底层逻辑.md) |
| 50 | [TensorFlow GradientTape 机制](50_TensorFlow GradientTape 机制.md) |
| 51 | [神经网络的可逆性 (Invertible Networks)](51_神经网络的可逆性 (Invertible Networks).md) |
| 52 | [深度平衡模型 (Deep Equilibrium Models)](52_深度平衡模型 (Deep Equilibrium Models).md) |
| 53 | [神经微分方程 (Neural ODEs) 基础](53_神经微分方程 (Neural ODEs) 基础.md) |
| 54 | [元学习 (Meta-Learning) 与 MAML 算法](54_元学习 (Meta-Learning) 与 MAML 算法.md) |
| 55 | [少样本学习 (Few-Shot Learning) 范式](55_少样本学习 (Few-Shot Learning) 范式.md) |
| 56 | [自监督学习 (Self-Supervised Learning) 概述](56_自监督学习 (Self-Supervised Learning) 概述.md) |
| 57 | [对比学习 (Contrastive Learning) 框架：SimCLR](57_对比学习 (Contrastive Learning) 框架：SimCLR.md) |
| 58 | [知识蒸馏 (Knowledge Distillation) 损失设计](58_知识蒸馏 (Knowledge Distillation) 损失设计.md) |
| 59 | [模型集成 (Ensembling) 的策略与收益](59_模型集成 (Ensembling) 的策略与收益.md) |
| 60 | [神经架构搜索 (NAS) 基础方法](60_神经架构搜索 (NAS) 基础方法.md) |
