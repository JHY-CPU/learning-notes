import os

titles = [
    "41_梯度裁剪 (Gradient Clipping) 防止爆炸",
    "42_学习率调度器：StepLR 与 MultiStepLR",
    "43_Cosine Annealing 余弦退火策略",
    "44_One Cycle Learning Rate Policy",
    "45_Warmup 预热阶段的必要性分析",
    "46_混合精度训练 (Mixed Precision) 原理",
    "47_损失缩放 (Loss Scaling) 技术细节",
    "48_动态计算图与静态计算图的对比",
    "49_PyTorch Autograd 引擎底层逻辑",
    "50_TensorFlow GradientTape 机制",
    "51_神经网络的可逆性 (Invertible Networks)",
    "52_深度平衡模型 (Deep Equilibrium Models)",
    "53_神经微分方程 (Neural ODEs) 基础",
    "54_元学习 (Meta-Learning) 与 MAML 算法",
    "55_少样本学习 (Few-Shot Learning) 范式",
    "56_自监督学习 (Self-Supervised Learning) 概述",
    "57_对比学习 (Contrastive Learning) 框架：SimCLR",
    "58_知识蒸馏 (Knowledge Distillation) 损失设计",
    "59_模型集成 (Ensembling) 的策略与收益",
    "60_神经架构搜索 (NAS) 基础方法"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
