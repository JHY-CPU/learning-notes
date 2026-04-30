# 64_Hough 变换与直线及圆检测

## 核心概念

- **Hough变换（霍夫变换）**：一种从图像中检测几何形状（直线、圆、椭圆）的特征提取技术，通过将图像空间中的点映射到参数空间中的"投票"来检测形状。
- **直线检测的Hough变换**：将笛卡尔坐标系中的直线 $y = kx + b$ 用极坐标表示 $\rho = x\cos\theta + y\sin\theta$，其中 $\rho$ 是原点到直线的距离，$\theta$ 是垂线与x轴的夹角。
- **投票过程**：图像空间中的每个边缘点对应参数空间中的一条正弦曲线。多个共线边缘点的曲线会在参数空间中的同一点相交。统计参数空间中各点的投票数，投票数超过阈值的参数对应检测到的直线。
- **概率Hough变换（Probabilistic Hough Transform）**：标准Hough变换的改进——只随机选取部分边缘点进行投票，大幅减少计算量，并能直接输出线段的端点。
- **圆检测的Hough变换**：圆的一般方程为 $(x - a)^2 + (y - b)^2 = r^2$，参数空间是三维 $(a, b, r)$。每个边缘点对应一个锥面。使用Hough梯度法可以降低维度。
- **累加器（Accumulator）**：参数空间的离散化网格，用于统计投票数。累加器的分辨率（bin大小）影响检测精度和计算量之间的平衡。

## 数学推导

**直线的极坐标表示：**
$$
\rho = x\cos\theta + y\sin\theta
$$

其中 $\rho \in [-d_{max}, d_{max}]$（$d_{max}$ 是图像对角线长度），$\theta \in [0, \pi)$。

**Hough变换直线检测算法：**
1. 对图像进行边缘检测（如Canny），得到边缘点集 $\{(x_i, y_i)\}$
2. 初始化累加器 $A(\rho, \theta) = 0$
3. 对每个边缘点 $(x_i, y_i)$，对每个 $\theta \in [0, \pi)$（离散化）：
   - 计算 $\rho = x_i\cos\theta + y_i\sin\theta$
   - $A(\rho, \theta) += 1$
4. 在累加器 $A$ 中寻找局部最大值（投票峰值）
5. 投票数超过阈值 $T$ 的 $(\rho_k, \theta_k)$ 对应检测到的直线

**圆的Hough变换（标准方法）：**
对于圆 $(x - a)^2 + (y - b)^2 = r^2$，每个边缘点 $(x_i, y_i)$ 对应3D参数空间中的锥体：
$$
(a - x_i)^2 + (b - y_i)^2 = r^2
$$

**圆的Hough梯度法（降低维度）：**
1. 用Sobel计算每个边缘点的梯度方向 $\phi$
2. 梯度方向指向圆心：圆心 $(a, b) = (x_i + r\cos\phi, y_i + r\sin\phi)$
3. 对每个边缘点，沿梯度方向对 $r$ 进行投票
4. 投票峰值对应检测到的圆

## 直观理解

Hough变换可以理解为"民主投票"的几何决策过程——每个边缘点"投票"支持所有可能通过它的直线。如果多个点共线，它们支持的"候选直线"集中在参数空间的某一点，形成投票峰值。

将直线用 $\rho-\theta$ 表示而非 $y = kx + b$ 的原因是：垂直线的斜率为无穷大，无法用 $k$ 表示。而 $\rho-\theta$ 表示对所有方向的直线都是统一的。

想象你在一个广场上看到散布在地面上的标记（边缘点），你需要判断哪些标记是在一条直线上。你可以拿一把尺子在每个标记上尝试每个角度——如果某条直线通过了多个标记，你就在这条直线的"得分卡"上加一分。最终，得分最高的直线就是检测结果。

## 代码示例

```python
import torch
import math

def hough_line_transform(edge_image, theta_res=180, rho_res=200, threshold=50):
    """标准Hough变换直线检测 (简化PyTorch实现)"""
    B, _, H, W = edge_image.shape
    diag = int(math.sqrt(H**2 + W**2)) + 1  # 最大rho
    
    # 参数空间离散化
    thetas = torch.linspace(0, math.pi, theta_res)
    rhos = torch.linspace(-diag, diag, rho_res * 2)
    
    # 累加器
    accumulator = torch.zeros(theta_res, rho_res * 2)
    
    # 获取边缘点坐标
    edge_indices = torch.nonzero(edge_image[0, 0] > 0)
    num_edges = edge_indices.shape[0]
    
    if num_edges == 0:
        return accumulator, thetas, rhos
    
    # 投票
    cos_theta = torch.cos(thetas)  # (theta_res)
    sin_theta = torch.sin(thetas)  # (theta_res)
    
    for idx in edge_indices:
        y, x = idx[0].float(), idx[1].float()
        # 计算所有theta下的rho
        rho_vals = x * cos_theta + y * sin_theta
        # 找到在rho离散化中的索引
        rho_indices = ((rho_vals + diag) * (rho_res * 2 - 1) / (2 * diag)).long()
        rho_indices = torch.clamp(rho_indices, 0, rho_res * 2 - 1)
        # 投票 (批量)
        for t in range(theta_res):
            accumulator[t, rho_indices[t]] += 1
    
    return accumulator, thetas, rhos

def extract_lines(accumulator, thetas, rhos, threshold):
    """从累加器中提取检测到的直线"""
    lines = []
    # 寻找局部最大值
    for t in range(1, accumulator.shape[0] - 1):
        for r in range(1, accumulator.shape[1] - 1):
            val = accumulator[t, r]
            if val > threshold:
                # 检查是否是局部最大值
                if val >= accumulator[t-1:t+2, r-1:r+2].max():
                    lines.append((thetas[t], rhos[r], val))
    return lines

# 模拟检测直线
def demo_hough():
    # 创建含有一条直线的图像
    img = torch.zeros(1, 1, 100, 100)
    for i in range(30, 70):
        img[0, 0, i, i + 20] = 1.0  # 对角线偏移的直线
    
    acc, thetas, rhos = hough_line_transform(img, theta_res=180, rho_res=100, threshold=10)
    lines = extract_lines(acc, thetas, rhos, threshold=30)
    
    print(f"检测到 {len(lines)} 条直线:")
    for theta, rho, votes in lines:
        print(f"  θ={theta*180/math.pi:.1f}°, ρ={rho:.1f}, 票数={votes:.0f}")
    print(f"累加器尺寸: {acc.shape}")

demo_hough()

# Hough圆检测 (简化)
def hough_circle_demo():
    """Hough圆检测的投票概念演示"""
    # 创建一个含圆的图像
    img = torch.zeros(1, 1, 100, 100)
    cx, cy, r = 50, 50, 20
    for angle in range(360):
        x = int(cx + r * math.cos(angle * math.pi / 180))
        y = int(cy + r * math.sin(angle * math.pi / 180))
        if 0 <= x < 100 and 0 <= y < 100:
            img[0, 0, y, x] = 1.0
    
    print(f"\n圆检测的Hough变换:")
    print(f"图像: {img.shape}, 真实圆: center=({cx},{cy}), radius={r}")
    print("参数空间为3D (a, b, r), 每个边缘点投出锥体")
    print("累加器峰值对应检测到的圆")

hough_circle_demo()

print("\nHough变换的优缺点:")
print("+ 对噪声鲁棒 (民主投票机制)")
print("+ 对遮挡鲁棒 (部分边缘也能检测)")
print("+ 可检测任意参数化形状")
print("- 计算量大 (尤其高维参数空间)")
print("- 参数选择敏感 (阈值, 累加器分辨率)")
```

## 深度学习关联

- **与深度学习方法的互补**：虽然基于学习的特征检测（如基于CNN的线检测、LSD直线检测等）在复杂场景中表现更好，但Hough变换由于其数学可解释性和无训练需求，在工业视觉检测、文档分析、车道线检测等场景中仍然广泛使用。
- **可微Hough变换**：近期工作（如Deep Hough Transform、HoughNet）将Hough变换嵌入到深度学习框架中，使其可微，让网络可以端到端地学习在Hough空间中的特征表示，结合了传统投票机制和深度学习的灵活性。
- **车道线检测中的应用**：Hough变换在自动驾驶的车道线检测中是最经典的方法——边缘检测后通过Hough变换提取车道线，再进行车道线过滤和跟踪。虽然现代方法更多使用基于分割的深度学习（如LaneNet、LaneATT），但Hough变换仍然是理解车道线和消失点几何关系的重要工具。
