import os

titles = [
    "41_神经辐射场 (NeRF) 基础原理",
    "42_3D Gaussian Splatting 渲染技术",
    "43_图像检索与哈希学习 (Hash Learning)",
    "44_零样本学习 (Zero-Shot Learning) 在 CV 中的应用",
    "45_自监督视觉预训练：MAE (Masked Autoencoders)",
    "46_DINOv2：无需标签的视觉特征提取",
    "47_CLIP 模型：图文对齐与多模态理解",
    "48_Stable Diffusion 在图像编辑中的应用",
    "49_ControlNet：精确控制生成图像结构",
    "50_Segment Anything Model (SAM) 架构分析",
    "51_视觉问答 (VQA) 系统设计",
    "52_图像描述 (Image Captioning) 生成技术",
    "53_OCR 光学字符识别与 CRNN 模型",
    "54_车牌识别与场景文本检测 (DBNet)",
    "55_医学图像分割中的 Dice Loss",
    "56_遥感图像中的旋转目标检测",
    "57_工业缺陷检测中的小样本问题",
    "58_自动驾驶中的多任务学习网络",
    "59_BEV (Bird's Eye View) 视角转换技术",
    "60_视觉 SLAM 与深度学习结合"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
