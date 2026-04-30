import os

titles = [
    "61_图像插值算法：双线性与双三次插值",
    "62_颜色空间转换：RGB, HSV, LAB 的差异",
    "63_边缘检测算子：Sobel, Canny 原理",
    "64_Hough 变换与直线及圆检测",
    "65_SIFT 特征提取与匹配",
    "66_ORB 特征点检测与描述子",
    "67_相机标定与内参及外参矩阵",
    "68_立体视觉与深度图生成",
    "69_全景拼接与图像配准",
    "70_视频目标跟踪：Siamese 网络"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
