# 36_光流估计 (Optical Flow) 与运动分析

## 核心概念

- **光流（Optical Flow）定义**：描述图像序列中像素在时间域上的运动向量场，即每个像素在连续帧之间的位移。光流估计的目标是计算每个像素的运动方向和速度。
- **亮度恒常假设（Brightness Constancy）**：光流估计的基本假设——同一点在相邻帧中的亮度（像素值）保持不变。$I(x,y,t) = I(x+dx, y+dy, t+dt)$。
- **Lucas-Kanade方法**：局部光流方法，假设在小邻域内所有像素具有相同的运动向量，通过最小二乘法求解光流方程。适合稀疏特征点的跟踪。
- **Horn-Schunck方法**：全局光流方法，在亮度恒常假设的基础上加入全局平滑约束，求解稠密光流场。通过变分法最小化能量函数。
- **深度学习光流（FlowNet/RAFT）**：FlowNet首次使用端到端CNN直接预测光流；RAFT（Recurrent All-Pairs Field Transforms）使用迭代优化在4D相关体上进行查找，达到SOTA精度。
- **光流的应用**：视频稳定、运动目标检测与跟踪、动作识别、视频插帧、深度估计（运动结构）等。

## 数学推导

**光流约束方程（由亮度恒常假设推导）：**
$$
I(x + dx, y + dy, t + dt) = I(x, y, t)
$$

对左侧进行一阶泰勒展开：
$$
I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt = I(x,y,t)
$$

化简得光流约束方程：
$$
\frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} + \frac{\partial I}{\partial t} = 0
$$

即：
$$
I_x u + I_y v + I_t = 0
$$

其中 $u = dx/dt$, $v = dy/dt$ 是光流的两个分量。

**孔径问题：** 光流约束方程包含两个未知数 $(u,v)$，但只有一个方程——这就是"孔径问题"。为了求解，需要额外的约束条件（局部邻域一致性或全局平滑性）。

**Horn-Schunck能量函数：**
$$
E = \iint \left[(I_x u + I_y v + I_t)^2 + \alpha^2 (|\nabla u|^2 + |\nabla v|^2)\right] dx dy
$$

第一项是光流约束，第二项是平滑约束（鼓励相邻像素具有相似的光流），$\alpha$ 控制平滑强度。

**RAFT的迭代优化：**
$$
v_{k+1} = v_k + \Delta v_k
$$

其中 $v_k$ 是第 $k$ 次迭代的光流估计，$\Delta v_k$ 是通过GRU循环单元预测的残差更新量。

## 直观理解

光流可以理解为"视频中像素的运动轨迹"。想象你在拍一段视频——一个行人在图像中从左向右走，该行人的每个像素在帧与帧之间都有一定的位移，这个位移向量就是光流。

亮度恒常假设是最基本的假设：它认为"[同一个物体点]在不同帧中的颜色/亮度应该是相同的"。这就像你在追踪一张白纸上的黑点——不管纸怎么移动，黑点始终是黑的。当然，这个假设在真实场景中经常被违反（光照变化、遮挡等），所以需要更复杂的模型来处理。

孔径问题类似于"通过一个小孔看运动的条纹"——你只能看到垂直方向的运动分量，看不到水平方向，因为小孔的限制让你失去了全局信息。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowNetS(nn.Module):
    """简化的 FlowNet (FlowNetSimple)"""
    def __init__(self):
        super().__init__()
        # 将两帧图像在通道维度拼接作为输入
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
        )
        # 预测光流
        self.predict_flow4 = nn.Conv2d(512, 2, 3, padding=1)
        # 上采样 + 细化
        self.deconv4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.predict_flow3 = nn.Conv2d(256 + 2, 2, 3, padding=1)

    def forward(self, img1, img2):
        # 拼接两帧
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
        x1 = self.conv1(x)   # 1/2
        x2 = self.conv2(x1)  # 1/4
        x3 = self.conv3(x2)  # 1/8
        x4 = self.conv4(x3)  # 1/16

        flow4 = self.predict_flow4(x4)  # 粗糙光流
        # 上采样光流
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=False)
        # 与编码器特征拼接
        up4 = self.deconv4(x4)
        flow3 = self.predict_flow3(torch.cat([up4, flow4_up], dim=1))

        # 上采样到原始尺寸
        flow = F.interpolate(flow3, scale_factor=4, mode='bilinear', align_corners=False)
        return flow  # (B, 2, H, W) 其中2表示(u, v)

# 光流的可视化 (将光流转为颜色编码)
def flow_to_rgb(flow):
    """将光流场转为RGB图像可视化"""
    B, _, H, W = flow.shape
    u = flow[:, 0]  # 水平分量
    v = flow[:, 1]  # 垂直分量
    
    mag = torch.sqrt(u**2 + v**2)  # 运动幅度
    ang = torch.atan2(v, u)  # 运动方向
    
    # HSV编码: H=方向, S=1.0, V=幅度
    h = (ang + torch.pi) / (2 * torch.pi)  # [0, 1]
    s = torch.ones_like(h)
    v = mag / (mag.max() + 1e-8)
    
    # HSV转RGB (简化)
    h = h * 6
    i = h.floor()
    f = h - i
    # ... 完整的HSV到RGB转换 (此处省略)
    return h.unsqueeze(1).repeat(1, 3, 1, 1)  # 简化返回

model = FlowNetS()
img1 = torch.randn(1, 3, 256, 256)
img2 = torch.randn(1, 3, 256, 256)
flow = model(img1, img2)
print(f"光流输出: {flow.shape}")  # (1, 2, 256, 256)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **视频理解的核心技术**：光流是视频分析的基础，被广泛用于行为识别（Two-Stream Networks以光流作为输入）、视频目标跟踪（光流引导的目标搜索）、视频分割等任务中。
- **从FlowNet到RAFT的发展**：深度学习光流从FlowNet（2015）的端到端回归，经过FlowNet 2.0（堆叠多个FlowNet）、PWC-Net（金字塔结构），发展到RAFT（2020）的迭代优化+4D相关体，精度不断提升。
- **光流与自监督学习**：光流估计可以作为自监督学习的代理任务——通过视频帧间的一致性作为监督信号，不需要人工标注光流真值。这为无监督/自监督视觉表示学习提供了重要的训练信号源。
