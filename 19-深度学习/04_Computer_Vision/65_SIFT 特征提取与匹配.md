# 65_SIFT 特征提取与匹配

## 核心概念

- **SIFT（Scale-Invariant Feature Transform）**：Lowe (1999, 2004) 提出的局部特征描述算法，对图像的缩放、旋转、光照变化和视角变化具有不变性，是计算机视觉最经典的特征提取方法之一。
- **尺度空间极值检测**：构建高斯金字塔（不同尺度的高斯模糊图像），在DoG（Difference of Gaussian）尺度空间中检测局部极值点作为候选关键点。DoG近似于尺度归一化的LoG（Laplacian of Gaussian）。
- **关键点精确定位**：对候选关键点进行亚像素精确定位（通过3D二次函数拟合），并去除低对比度点和边缘响应点（使用Hessian矩阵的主曲率比）以提高稳定性。
- **方向分配**：基于关键点邻域像素的梯度方向直方图（36个bin），分配一个或多个主方向。这实现了旋转不变性。
- **关键点描述子**：在关键点周围的 $16\times16$ 邻域内，划分为 $4\times4$ 个子区域，每个子区域计算8个方向的梯度直方图，形成 $4 \times 4 \times 8 = 128$ 维的特征向量。
- **特征匹配**：使用最近邻距离比（NNDR，Nearest Neighbor Distance Ratio）进行匹配——如果最近邻距离显著小于次近邻距离（比率<阈值，如0.7），则认为匹配可靠。
- **SIFT的专利与开源替代**：SIFT具有专利保护（2020年过期），现在可以自由使用。开源的替代方案包括A-KAZE、RootSIFT（对SIFT描述子应用Hellinger距离）等。

## 数学推导

**高斯金字塔与DoG：**
$$
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
$$
$$
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
$$

其中 $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2 + y^2)/(2\sigma^2)}$。

**关键点定位（3D二次函数拟合）：**
将DoG函数 $D(x)$ 在关键点处进行泰勒展开：
$$
D(x) = D + \frac{\partial D^T}{\partial x} x + \frac{1}{2} x^T \frac{\partial^2 D}{\partial x^2} x
$$

其中 $x = (x, y, \sigma)^T$。求解极值点偏移：
$$
\hat{x} = -\frac{\partial^2 D^{-1}}{\partial x^2} \frac{\partial D}{\partial x}
$$

如果 $|\hat{x}| > 0.5$ 在任何维度，说明极值点更靠近相邻像素，需要插值。

**去除边缘响应：**
使用Hessian矩阵 $H = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$ 的主曲率比：
$$
\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}
$$

其中 $r = 10$（默认阈值）。拒绝曲率比超过阈值的点（边缘上的点）。

**SIFT描述子的128维向量：**
在 $4\times4$ 网格的每个子区域中统计8个方向的梯度直方图，归一化后得到：
$$
v = [h_{1,1,1}, \dots, h_{1,1,8}, h_{1,2,1}, \dots, h_{4,4,8}]
$$

然后进行向量归一化和截断（将大于0.2的梯度值截断为0.2），再归一化，以增强对光照变化的鲁棒性。

## 直观理解

SIFT可以看作是"图像中的地标"——它从图像中挑选出一些独特、可重复检测的点（如角点、斑点），并为每个点生成一个"指纹"（128维描述子）。这些指纹对图像的放大缩小、旋转、光照变化都不敏感。

想象你在不同的天气、不同的角度拍摄同一座山——山上的某个特殊形状的岩石（SIFT关键点）无论在照片中被放大还是缩小、旋转还是倾斜，你都能认出这是"同一块岩石"。SIFT描述子就像给这块岩石测了128个维度的"特征向量"——颜色分布、纹理走向、周围环境等，使得不同图像中对应的岩石可以匹配上。

## 代码示例

```python
import torch
import torch.nn.functional as F
import math

def build_gaussian_pyramid(image, num_octaves=4, num_scales=5):
    """构建高斯金字塔 (简化)"""
    # image: (1, 1, H, W)
    pyramid = []
    sigma = 1.6
    k = 2.0 ** (1.0 / (num_scales - 1))
    
    for o in range(num_octaves):
        octave_images = []
        for s in range(num_scales):
            current_sigma = sigma * (k ** s)
            kernel_size = int(2 * (3 * current_sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            # 简化的高斯模糊 (使用平均池化近似)
            blurred = F.avg_pool2d(image, kernel_size, stride=1, 
                                    padding=kernel_size//2)
            octave_images.append(blurred)
        pyramid.append(octave_images)
        # 下采样到下一八度
        image = F.interpolate(image, scale_factor=0.5, mode='bilinear')
    
    return pyramid

def compute_dog(pyramid):
    """计算DoG (Difference of Gaussian)"""
    dog_pyramid = []
    for octave in pyramid:
        dog_images = []
        for i in range(len(octave) - 1):
            dog = octave[i+1] - octave[i]
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    return dog_pyramid

def find_keypoints(dog_pyramid, contrast_threshold=0.04):
    """查找关键点 (简化: 局部极值检测)"""
    keypoints = []
    for o, dog_octave in enumerate(dog_pyramid):
        for i in range(1, len(dog_octave) - 1):
            current = dog_octave[i][0, 0]
            H, W = current.shape
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    val = current[y, x]
                    # 检查是否是3x3x3邻域中的极值
                    neighbors = []
                    for d in [-1, 0, 1]:
                        patch = dog_octave[i + d][0, 0, y-1:y+2, x-1:x+2]
                        neighbors.extend(patch.flatten().tolist())
                    if val > contrast_threshold and val == max(neighbors):
                        keypoints.append((o, i, y, x, val))
                    elif val < -contrast_threshold and val == min(neighbors):
                        keypoints.append((o, i, y, x, val))
    return keypoints

def compute_sift_descriptor(image, keypoint, num_bins=8, grid_size=4):
    """SIFT描述子计算 (简化)"""
    # 获取关键点邻域的梯度
    y, x = int(keypoint[2]), int(keypoint[3])
    H, W = image.shape
    
    # 简化的梯度计算
    if y < 8 or y >= H - 8 or x < 8 or x >= W - 8:
        return None
    
    patch = image[y-8:y+8, x-8:x+8]
    gy, gx = torch.gradient(patch, dim=(0, 1))
    magnitude = torch.sqrt(gx**2 + gy**2)
    direction = torch.atan2(gy, gx) % (2 * math.pi)
    
    # 4x4子区域, 每个区域8方向直方图
    descriptor = []
    cell_size = 4
    for row in range(grid_size):
        for col in range(grid_size):
            hist = torch.zeros(num_bins)
            for dy in range(cell_size):
                for dx in range(cell_size):
                    py = row * cell_size + dy
                    px = col * cell_size + dx
                    mag = magnitude[py, px]
                    ang = direction[py, px]
                    bin_idx = int(ang / (2 * math.pi) * num_bins) % num_bins
                    hist[bin_idx] += mag
            descriptor.extend(hist.tolist())
    
    # 归一化和截断
    desc = torch.tensor(descriptor)
    desc = desc / (desc.norm() + 1e-10)
    desc = torch.clamp(desc, max=0.2)
    desc = desc / (desc.norm() + 1e-10)
    return desc

def match_sift(desc1, desc2, ratio_thresh=0.7):
    """SIFT特征匹配 (NNDR)"""
    # desc1: (N1, 128), desc2: (N2, 128)
    dists = torch.cdist(desc1, desc2)  # (N1, N2)
    
    matches = []
    for i in range(len(desc1)):
        sorted_dists, indices = torch.sort(dists[i])
        if sorted_dists[0] / (sorted_dists[1] + 1e-10) < ratio_thresh:
            matches.append((i, indices[0].item(), sorted_dists[0].item()))
    
    return matches

# 演示
img = torch.randn(1, 1, 128, 128)
pyramid = build_gaussian_pyramid(img)
dog = compute_dog(pyramid)
kps = find_keypoints(dog, contrast_threshold=0.05)
print(f"检测到 {len(kps)} 个关键点")

if kps:
    desc = compute_sift_descriptor(img[0, 0], kps[0])
    if desc is not None:
        print(f"SIFT描述子维度: {len(desc)}")
        print(f"描述子范数: {desc.norm():.4f}")

# 特征匹配演示
n1, n2 = 50, 80
desc1 = torch.randn(n1, 128)
desc2 = torch.randn(n2, 128)
matches = match_sift(desc1, desc2, ratio_thresh=0.7)
print(f"\nNNDR匹配: {len(matches)}/{n1} 个匹配对")
```

## 深度学习关联

- **SIFT与深度学习特征的对比**：SIFT是"手工设计特征"的巅峰之作，其不变性（尺度、旋转、光照）是通过精心设计的数学方式实现的。深度学习特征（如DINOv2、CLIP的特征）是通过大量数据学习得到的，具有更强的语义表示能力，但在某些几何变换不变性方面可能不如SIFT。
- **SuperPoint + SuperGlue（学习的特征替代）**：深度学习特征点检测方法SuperPoint（自监督训练的CNN特征点）和匹配方法SuperGlue（图神经网络匹配）正在替代SIFT在SLAM、SfM（运动恢复结构）中的应用，尤其在纹理稀疏和光照变化剧烈的场景中表现更好。
- **SIFT在现代视觉中的延续**：尽管深度学习已占据主导，SIFT仍在很多实际系统中使用（如全景拼接、3D重建的SfM管线、卫星图像配准等）。其关键思想（尺度空间、方向直方图描述子、最近邻匹配）也被大量深度学习特征方法借鉴和改进。
