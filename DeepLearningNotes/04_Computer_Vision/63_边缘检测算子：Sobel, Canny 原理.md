# 63_边缘检测算子：Sobel, Canny 原理

## 核心概念

- **图像边缘**：图像中像素值发生剧烈变化的区域，对应物体的轮廓、纹理边界或光照变化。边缘检测是计算机视觉中最底层的特征提取操作之一。
- **图像梯度**：边缘可以通过计算图像的一阶导数（梯度）来检测。梯度幅值大的位置对应边缘。梯度方向垂直于边缘方向。
- **Sobel算子**：基于离散微分的边缘检测算子，使用两个 $3\times3$ 卷积核分别计算水平和垂直方向的梯度近似值。计算简单、速度快，但对噪声敏感。
- **Canny边缘检测**：John Canny (1986) 提出的多阶段边缘检测算法，被认为是边缘检测的黄金标准。具有低错误率、良好定位和单响应三个优点。
- **Canny的四阶段流程**：**(1)** 高斯滤波去噪；**(2)** 计算梯度幅值和方向（使用Sobel）；**(3)** 非极大值抑制（NMS）——只保留梯度方向上的局部最大值，使边缘细化到单像素宽度；**(4)** 双阈值检测（高阈值和低阈值）——确定强边缘、弱边缘，抑制非边缘。
- **滞后阈值跟踪（Hysteresis Thresholding）**：双阈值检测后，只有与强边缘相连的弱边缘才被保留，其他弱边缘被抑制。这有助于连接断裂的边缘。

## 数学推导

**Sobel算子：**

水平梯度核（检测垂直边缘）：
$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I
$$

垂直梯度核（检测水平边缘）：
$$
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I
$$

梯度幅值和方向：
$$
G = \sqrt{G_x^2 + G_y^2}
$$
$$
\Theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

**Canny边缘检测的非极大值抑制：**
对每个像素 $(i,j)$，比较其梯度幅值 $G(i,j)$ 与梯度方向 $\Theta(i,j)$ 上相邻两个像素的幅值。若 $G(i,j)$ 不是局部最大值，则置零。

梯度方向量化为4个扇区：水平（0°）、对角（45°）、垂直（90°）、反对角（135°）。

**Canny的双阈值检测：**
$$
\text{若 } G(i,j) > T_{high} \Rightarrow \text{强边缘}
$$
$$
\text{若 } T_{low} < G(i,j) \le T_{high} \Rightarrow \text{弱边缘}
$$
$$
\text{若 } G(i,j) \le T_{low} \Rightarrow \text{抑制}
$$

滞后阈值跟踪：弱边缘只有在8邻域内存在强边缘时才被保留。

## 直观理解

Sobel算子在每个像素周围计算"亮度变化的速度和方向"——水平核 $G_x$ 像是在问"左右两边的亮度差是多少"，垂直核 $G_y$ 在问"上下两边的亮度差是多少"。两个核的响应组合起来就得到了边缘的强度和方向。

Canny算法像一个严格的"考官"——先模糊化（高斯滤波）去除噪声干扰，然后计算梯度找到所有可能的边缘位置。之后再通过"非极大值抑制"收紧边缘（确保边缘只有一个像素宽），最后用"双阈值"做质量筛选——高分像素直接通过（强边缘），低分像素直接淘汰，中间分数像素只有"认识"（连接）到高分像素才能通过。这就像一个先海选、再淘汰、最后靠关系网来决定的选拔过程。

## 代码示例

```python
import torch
import torch.nn.functional as F

def sobel_edge_detection(image):
    """Sobel边缘检测 (PyTorch实现)"""
    # image: (B, 1, H, W)
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = torch.atan2(grad_y, grad_x)
    
    return magnitude, direction, grad_x, grad_y

def canny_edge_detection(image, low_threshold=0.1, high_threshold=0.3):
    """简化版Canny边缘检测 (PyTorch)"""
    # 1. 高斯滤波
    kernel_size = 5
    sigma = 1.0
    gaussian = torch.tensor([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2],
    ], dtype=torch.float32).view(1, 1, 5, 5) / 159.0
    
    blurred = F.conv2d(image, gaussian, padding=2)
    
    # 2. Sobel梯度
    magnitude, direction, grad_x, grad_y = sobel_edge_detection(blurred)
    
    # 3. 非极大值抑制 (简化实现)
    # 将方向量化为0, 45, 90, 135度
    angle = direction * 180.0 / torch.pi
    angle[angle < 0] += 180.0
    
    nms = magnitude.clone()
    B, _, H, W = magnitude.shape
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # 根据梯度方向比较相邻像素
            if (0 <= angle[0, 0, i, j] < 22.5) or (157.5 <= angle[0, 0, i, j] <= 180):
                neighbors = (magnitude[0, 0, i, j-1], magnitude[0, 0, i, j+1])
            elif 22.5 <= angle[0, 0, i, j] < 67.5:
                neighbors = (magnitude[0, 0, i-1, j+1], magnitude[0, 0, i+1, j-1])
            elif 67.5 <= angle[0, 0, i, j] < 112.5:
                neighbors = (magnitude[0, 0, i-1, j], magnitude[0, 0, i+1, j])
            else:
                neighbors = (magnitude[0, 0, i-1, j-1], magnitude[0, 0, i+1, j+1])
            
            if magnitude[0, 0, i, j] < max(neighbors):
                nms[0, 0, i, j] = 0.0
    
    # 4. 双阈值检测
    strong = nms > high_threshold
    weak = (nms > low_threshold) & (nms <= high_threshold)
    
    # 5. 滞后阈值跟踪 (简化: 8邻域搜索)
    edges = strong.float()
    # 标记与强边缘相连的弱边缘
    kernel = torch.ones(1, 1, 3, 3)
    for _ in range(3):  # 多次传播
        connected = F.conv2d(edges, kernel, padding=1) > 0
        new_edges = weak & connected
        edges = edges | new_edges.float()
    
    return edges

# 测试
image = torch.zeros(1, 1, 64, 64)
# 画一个简单的方形
image[0, 0, 20:40, 20:40] = 1.0

mag, direction, gx, gy = sobel_edge_detection(image)
print(f"Sobel梯度幅值范围: [{mag.min().item():.4f}, {mag.max().item():.4f}]")
print(f"梯度方向范围: [{direction.min().item():.2f}, {direction.max().item():.2f}]")

# Canny边缘检测
edges = canny_edge_detection(image, low_threshold=0.1, high_threshold=0.3)
print(f"Canny边缘检测输出: {edges.shape}")

# Sobel vs Canny的可视化对比
print("\nSobel vs Canny:")
print("- Sobel: 速度快, 边缘较粗, 对噪声敏感")
print("- Canny: 多阶段优化, 边缘细(单像素), 抗噪更强")
```

## 深度学习关联

- **深度学习中的边缘检测**：传统的Sobel和Canny算子为后来的深度学习边缘检测方法（HED、RCF、BDCN）提供了理论基础和基线比较标准。HED（Holistically-Nested Edge Detection）使用多尺度侧输出层在VGGNet中检测边缘，并在多个尺度上进行融合。
- **作为数据预处理和损失函数**：Sobel算子因其可微性被直接嵌入到CNN中用于计算梯度信息。例如，在图像超分辨率重建中，图像的梯度/边缘信息可以作为额外的监督信号（边缘损失）；在图像风格迁移中，Sobel梯度被用于计算纹理损失。
- **可学习的边缘检测**：现代方法用CNN学习边缘检测，超越了手工设计的Sobel/Canny的局限性（固定尺度、对光照和噪声敏感）。但Sobel/Canny作为快速、无参数、无需训练的预处理步骤，在传统图像处理流水线和嵌入式系统中仍然广泛使用。
