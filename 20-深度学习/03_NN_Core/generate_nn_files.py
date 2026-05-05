import os

titles = [
    "01_感知机模型与逻辑门实现",
    "02_多层感知机 (MLP) 的万能近似定理",
    "03_前向传播的计算图表示",
    "04_反向传播算法：链式法则的递归应用",
    "05_全连接层梯度推导：权重与偏置",
    "06_Sigmoid 激活函数与梯度消失问题",
    "07_Tanh 激活函数的零中心特性",
    "08_ReLU 及其变体 (Leaky ReLU, PReLU)",
    "09_GELU 与 Swish：现代 Transformer 的首选",
    "10_Softmax 函数推导与数值稳定性 (Log-Sum-Exp)",
    "11_均方误差 (MSE) 损失函数的梯度特性",
    "12_交叉熵损失 (Cross-Entropy) 的导数简化形式",
    "13_Huber Loss 对异常值的鲁棒性",
    "14_Contrastive Loss 与相似度学习",
    "15_Triplet Loss 与锚点选择策略",
    "16_SGD 优化器：随机性与逃逸局部极小值",
    "17_Momentum 动量项的物理类比与公式",
    "18_Nesterov Accelerated Gradient (NAG) 的前瞻性",
    "19_AdaGrad：自适应学习率与稀疏特征",
    "20_RMSProp：解决 AdaGrad 学习率单调递减",
    "21_Adam 优化器：一阶与二阶矩估计",
    "22_AdamW：解耦权重衰减 (Weight Decay)",
    "23_LAMB 与 LARS：大 Batch 训练优化器",
    "24_Xavier 初始化与方差保持推导",
    "25_He 初始化与 ReLU 的适配性",
    "26_正交初始化 (Orthogonal Initialization) 在 RNN 中的应用",
    "27_L2 正则化与权重衰减的等价性证明",
    "28_L1 正则化与稀疏解的几何解释",
    "29_Early Stopping 作为隐式正则化",
    "30_Dropout：集成学习的视角与 Inference 缩放",
    "31_Data Augmentation 数据增强的正则化效果",
    "32_Label Smoothing 标签平滑的原理与实现",
    "33_Batch Normalization：前向与反向传播推导",
    "34_Layer Normalization 与 BatchNorm 的对比",
    "35_Instance Normalization 与 Style Transfer",
    "36_Group Normalization：小 Batch 下的替代方案",
    "37_Residual Connection 残差连接的梯度流动分析",
    "38_DenseNet 密集连接与特征复用",
    "39_Neural Tangent Kernel (NTK) 理论基础",
    "40_损失曲面 (Loss Landscape) 可视化与分析"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
