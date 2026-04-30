import os

titles = [
    "31_Inception Score (IS) 的计算逻辑",
    "32_Precision and Recall for GANs",
    "33_流形假设与生成模型的关系",
    "34_能量基模型 (EBM) 基础",
    "35_扩散模型的 SDE 视角解释",
    "36_Consistency Models：一步生成技术",
    "37_Latent Consistency Models (LCM)",
    "38_Guidance Scale：分类器引导与无分类器引导",
    "39_Cross-Attention 在文本控制中的作用",
    "40_生成式 AI 的版权与伦理问题",
    "41_Deepfake 检测技术与防御",
    "42_提示词工程 (Prompt Engineering) 技巧",
    "43_Negative Prompt 的实现机制",
    "44_Inpainting 与 Outpainting 算法原理",
    "45_Model Merging 模型合并技术",
    "46_Distilled Diffusion 蒸馏加速",
    "47_Autoregressive vs Diffusion 生成范式对比",
    "48_Masked Generative Modeling (MAGVIT)",
    "49_多模态大模型 (LMM) 的生成能力",
    "50_世界模型 (World Models) 与生成式 AI"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
