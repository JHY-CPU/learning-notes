# 62_颜色空间转换：RGB, HSV, LAB 的差异

## 核心概念

- **RGB颜色空间**：基于三原色（红Red、绿Green、蓝Blue）的加色模型，是计算机图像的标准表示，每种颜色由三个通道的强度值（0-255）组合而成。
- **HSV颜色空间**：将颜色分解为色相（Hue, 色调）、饱和度（Saturation, 色彩纯度）和明度（Value, 亮度）。更接近人类对颜色的感知方式。
- **LAB颜色空间**：国际照明委员会（CIE）定义的感知均匀颜色空间——L表示亮度（Lightness），A表示绿到红，B表示蓝到黄。欧氏距离在LAB空间中近似颜色感知差异。
- **RGB到HSV的转换**：根据RGB三通道的最大值和最小值计算H、S、V值。H呈环状（0°-360°），S和V在[0,1]范围。
- **RGB到LAB的转换**：需要两步转换——先通过非线性变换（gamma校正）从RGB到XYZ，再从XYZ到LAB。XYZ是中间设备无关的颜色空间。
- **各空间的适用场景**：RGB适合显示和存储；HSV适合图像分割和颜色过滤（基于色调筛选目标）；LAB适合颜色差异度量（如颜色迁移、图像检索中的颜色相似度）。

## 数学推导

**RGB到HSV的转换：**
设 $R, G, B \in [0, 1]$，$max = \max(R,G,B)$，$min = \min(R,G,B)$，$\Delta = max - min$：

$$
V = max
$$
$$
S = \begin{cases}
0 & max = 0 \\
\frac{\Delta}{max} & otherwise
\end{cases}
$$
$$
H = \begin{cases}
0 & \Delta = 0 \\
60^\circ \times \left(\frac{G-B}{\Delta} \mod 6\right) & max = R \\
60^\circ \times \left(\frac{B-R}{\Delta} + 2\right) & max = G \\
60^\circ \times \left(\frac{R-G}{\Delta} + 4\right) & max = B
\end{cases}
$$

**RGB到LAB的转换：**

步骤1：RGB到XYZ（线性化 + 线性变换）：
$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = M \cdot \begin{bmatrix} f(R) \\ f(G) \\ f(B) \end{bmatrix}
$$

其中 $f$ 是gamma逆校正，$M$ 是与色域相关的3×3矩阵（如sRGB标准）。

步骤2：XYZ到LAB：
$$
L^* = 116 \cdot g\left(\frac{Y}{Y_n}\right) - 16
$$
$$
a^* = 500 \cdot \left[g\left(\frac{X}{X_n}\right) - g\left(\frac{Y}{Y_n}\right)\right]
$$
$$
b^* = 200 \cdot \left[g\left(\frac{Y}{Y_n}\right) - g\left(\frac{Z}{Z_n}\right)\right]
$$

其中 $X_n, Y_n, Z_n$ 是参考白点的XYZ值。
$$
g(t) = \begin{cases}
t^{1/3} & t > (6/29)^3 \\
\frac{t}{3 \times (6/29)^2} + \frac{4}{29} & otherwise
\end{cases}
$$

**LAB空间的色差公式（CIE76）：**
$$
\Delta E_{ab}^* = \sqrt{(L_1^* - L_2^*)^2 + (a_1^* - a_2^*)^2 + (b_1^* - b_2^*)^2}
$$

$\Delta E < 1$ 的色差被认为人眼不可察觉。

## 直观理解

RGB像是一个"电视屏幕"的工作方式——通过红绿蓝三色灯泡的亮度组合呈现所有颜色。这种方式非常适合显示设备，但不适合"思考颜色"——当你看到橙色时，不会去想"这应该是230的红加150的绿"，而更自然地想"这是30度的色相，饱和度较高，亮度中等"——这就是HSV的思考方式。

LAB空间则像是一个"颜色尺子"——L轴衡量亮度（从黑到白），A轴衡量从绿到红的程度，B轴衡量从蓝到黄的程度。这个空间的神奇之处在于：两个颜色在LAB空间中的直线距离大致等于人眼感知到的色差。这使得LAB成为颜色科学和质量控制的工业标准。

## 代码示例

```python
import torch
import numpy as np

def rgb_to_hsv(rgb):
    """RGB到HSV转换 (PyTorch实现)"""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    max_val, _ = torch.max(rgb, dim=1, keepdim=True)
    min_val, _ = torch.min(rgb, dim=1, keepdim=True)
    delta = max_val - min_val + 1e-8
    
    # V
    v = max_val
    # S
    s = delta / (max_val + 1e-8)
    
    # H
    h = torch.zeros_like(r)
    mask = (max_val == r) & (delta > 0)
    h[mask] = (60 * ((g - b) / delta) % 360)[mask]
    mask = (max_val == g) & (delta > 0)
    h[mask] = (60 * ((b - r) / delta) + 120)[mask]
    mask = (max_val == b) & (delta > 0)
    h[mask] = (60 * ((r - g) / delta) + 240)[mask]
    h = (h + 360) % 360
    
    return torch.cat([h/360.0, s, v], dim=1)

# HSV在图像分割中的应用
def color_segmentation_hsv(image_rgb, hue_target, hue_range=0.1, sat_min=0.1):
    """基于色调的图像分割"""
    hsv = rgb_to_hsv(image_rgb)
    h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    
    # 选择目标色调范围内的像素 (且饱和度足够高)
    hue_diff = torch.abs(h - hue_target)
    mask = (hue_diff < hue_range) & (s > sat_min)
    return mask.float()

# LAB色差计算
def rgb_to_lab(rgb):
    """简化的RGB到LAB转换"""
    # sRGB到线性
    linear = torch.where(rgb > 0.04045, 
                         ((rgb + 0.055) / 1.055) ** 2.4,
                         rgb / 12.92)
    # 线性RGB到XYZ (sRGB D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]).to(rgb.device)
    
    B, C, H, W = rgb.shape
    linear_perm = linear.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = linear_perm @ M.T
    xyz = xyz.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    
    # XYZ到LAB
    X, Y, Z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    X_n, Y_n, Z_n = 0.950456, 1.0, 1.088754
    
    def f(t):
        delta = 6.0/29.0
        return torch.where(t > delta**3, t**(1.0/3.0), t/(3*delta**2) + 4.0/29.0)
    
    L = 116 * f(Y/Y_n) - 16
    a = 500 * (f(X/X_n) - f(Y/Y_n))
    b = 200 * (f(Y/Y_n) - f(Z/Z_n))
    
    return torch.cat([L, a, b], dim=1)

def delta_e(lab1, lab2):
    """计算LAB色差 ΔE"""
    diff = lab1 - lab2
    return torch.sqrt((diff ** 2).sum(dim=1, keepdim=True))

# 演示
rgb = torch.randn(1, 3, 4, 4)
hsv_img = rgb_to_hsv(rgb)
print(f"HSV: {hsv_img.shape}")

# 绿色目标分割
green_mask = color_segmentation_hsv(rgb, hue_target=120/360, hue_range=30/360)
print(f"绿色分割掩码: {green_mask.shape}")

# 颜色迁移: LAB空间中的操作
lab1 = rgb_to_lab(torch.ones(1, 3, 1, 1))  # 白色
lab2 = rgb_to_lab(torch.zeros(1, 3, 1, 1))  # 黑色
print(f"黑白之间的ΔE: {delta_e(lab1, lab2).item():.2f}")

print("\n各颜色空间的特性总结:")
print("- RGB: 加色模型, 适合显示设备")
print("- HSV: 色相-饱和度-明度, 适合颜色选择和分割")
print("- LAB: 感知均匀, 色差度量标准")
```

## 深度学习关联

- **颜色增强与不变性**：在训练数据增强中，HSV空间的随机变换（色相抖动、饱和度调整）比RGB空间的直接扰动更自然，不会破坏图像的色调一致性。许多分类网络（如EfficientNet、ResNet）的训练流程中包含HSV增强。
- **LAB空间在颜色迁移中的应用**：风格迁移（Style Transfer）和图像上色（Image Colorization）中很多方法在LAB空间操作——将L通道（亮度结构）和AB通道（颜色信息）分开处理，使模型只需要预测两个颜色通道的值即可生成彩色图像。
- **颜色空间作为输入预处理**：某些任务中，将RGB图像转换到HSV或LAB空间可以提高模型性能。例如，在图像分割中同时使用RGB和HSV作为输入通道（6通道输入）可以提供互补的颜色信息；在夜景图像增强中，LAB空间的L通道处理可独立于颜色偏差。
