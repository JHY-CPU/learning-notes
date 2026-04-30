import os

titles = [
    "26_矩阵求导术：标量对向量求导",
    "27_矩阵求导术：向量对向量求导",
    "28_链式法则在计算图中的自动微分实现",
    "29_高维空间中的距离度量：欧氏距离与余弦相似度",
    "30_正交投影与最小二乘法的几何解释",
    "31_瑞利商 (Rayleigh Quotient) 与 PCA 推导",
    "32_马氏距离 (Mahalanobis Distance) 与协方差归一化",
    "33_Jensen 不等式及其在 EM 算法中的应用",
    "34_蒙特卡洛积分与期望估计",
    "35_重要性采样 (Importance Sampling) 原理",
    "36_变分推断 (Variational Inference) 基础",
    "37_指数族分布的通用形式与自然参数",
    "38_共轭先验与后验分布的解析解",
    "39_假设检验与 p-value 的本质",
    "40_置信区间与贝叶斯可信区间",
    "41_大数定律在模型评估中的体现",
    "42_傅里叶变换与频域分析基础",
    "43_卷积定理与频域滤波",
    "44_拉普拉斯变换与系统稳定性",
    "45_图论基础：邻接矩阵与拉普拉斯矩阵",
    "46_谱聚类与图切分 (Graph Cut)",
    "47_信息瓶颈理论 (Information Bottleneck)",
    "48_费雪信息矩阵 (Fisher Information Matrix)",
    "49_Cramer-Rao 下界与参数估计极限",
    "50_优化 landscape：鞍点与局部极小值"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
