# 70_视频目标跟踪：Siamese 网络

## 核心概念

- **视频目标跟踪 (Visual Object Tracking)**：在视频序列的第一帧中给定目标边界框，在后续每一帧中预测目标的位置和尺度。跟踪器需要在目标外观变化、遮挡、背景干扰等挑战下保持稳定。
- **Siamese 网络架构**：双分支共享权重的神经网络结构。两个分支分别处理模板图像（第一帧的目标）和搜索图像（当前帧），通过互相关计算相似度图，响应峰值位置即为目标新位置。
- **全卷积 Siamese 网络 (SiamFC)**：Bertinetto 等人在 2016 年提出的开创性工作。使用全卷积网络提取模板和搜索区域的特征，在特征域计算互相关，保持空间信息的平移不变性，输出密集响应的得分图。
- **区域提议网络 (RPN) 在跟踪中的应用**：SiamRPN 将目标检测中的 RPN 引入 Siamese 跟踪，同时输出分类分支（前景/背景）和回归分支（边界框调整），实现更精确的目标定位和尺度估计。
- **在线更新 vs. 离线训练**：传统跟踪器（如 KCF、MDNet）在跟踪过程中在线更新模型以适应目标外观变化，但存在漂移风险。Siamese 跟踪器完全离线训练、在线仅前向推理，不更新模型，避免了漂移问题。
- **判别式 Siamese 网络 (DaSiamRPN)**：通过数据增强和困难负样本挖掘提升 Siamese 跟踪器的判别能力，引入干扰感知模块在搜索区域中抑制相似物体的响应。

## 数学推导

**Siamese 跟踪的相似度学习**：
$$
f(z, x) = \varphi(z) \star \varphi(x) + b \cdot \mathbf{1}
$$

其中 $\varphi$ 是共享权重的卷积特征提取网络，$\star$ 表示互相关操作，$z$ 为模板图像（目标），$x$ 为搜索区域图像。输出 $f(z, x)$ 是一个二维得分图，峰值位置对应目标中心的相对位移。

**逻辑回归损失**（SiamFC 的训练损失）：
$$
\min_\theta \frac{1}{|\mathcal{D}|} \sum_{(z,x,y)\in\mathcal{D}} \ell(y, f(z,x;\theta))
$$

其中 $\ell$ 为逐像素的逻辑损失（二元交叉熵）：
$$
\ell(y, v) = \log(1 + \exp(-yv))
$$

$y \in \{-1, +1\}$ 为标注的对应位置标签（正样本为距目标中心 $R$ 半径内的位置），$v$ 为得分图上该位置的响应值。

**SiamRPN 的边界框回归**：
$$
\delta[0] = \frac{T_x - A_x}{A_w}, \quad
\delta[1] = \frac{T_y - A_y}{A_h}, \quad
\delta[2] = \ln\frac{T_w}{A_w}, \quad
\delta[3] = \ln\frac{T_h}{A_h}
$$

其中 $(A_x, A_y, A_w, A_h)$ 为锚点框，$(T_x, T_y, T_w, T_h)$ 为目标真实框，$(\delta[0], \delta[1], \delta[2], \delta[3])$ 为回归目标。

## 直观理解

- **"找不同"的变体**：Siamese 跟踪器的工作方式类似于"找不同"游戏。第一帧给定目标模板，相当于告诉网络"就是找这个东西"。在后续帧中，它在搜索区域中逐位置比对，找到与模板最相似的位置。共享权重的双分支结构确保模板和搜索区域使用相同的"眼睛"观察。
- **模板匹配的进化版**：传统的模板匹配在原始像素空间用滑动窗口+归一化互相关搜索目标，对目标外观变化非常敏感。Siamese 跟踪器相当于在深度特征空间中进行模板匹配，特征提取器经过大量数据的离线训练，能够对光照、形变、部分遮挡等因素保持鲁棒。
- **无更新策略的双刃剑**：Siamese 跟踪器不更新模板，这意味着它不会"忘记"目标的初始外观，但也无法适应目标的渐进式变化（如视角旋转 90 度后的完全不同的外观）。这解释了为什么 Siamese 跟踪器在短期跟踪中非常鲁棒，但在长期跟踪中可能因外观剧变而丢失目标。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class SiamFC(nn.Module):
    """简化的全卷积 Siamese 跟踪器"""
    def __init__(self):
        super().__init__()
        # 特征提取骨干网络（简化版 AlexNet）
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1),
            nn.ReLU(),
        )
    
    def forward(self, z, x):
        """z: 模板图像 [B, 3, 127, 127], x: 搜索图像 [B, 3, 255, 255]"""
        z_feat = self.features(z)   # [B, 256, 6, 6]
        x_feat = self.features(x)   # [B, 256, 22, 22]
        # 互相关
        return F.conv2d(x_feat, z_feat)  # [B, 1, 17, 17]

class SimpleTracker:
    """简化的目标跟踪器"""
    def __init__(self, model_path=None):
        self.model = SiamFC()
        self.model.eval()
        self.template = None
        self.target_pos = None
        self.target_sz = None
        self.score_size = 17
        self.stride = 8
        self.context = 0.5
        
        # 生成 cosine window（抑制边缘响应）
        self.hann = np.outer(
            np.hanning(self.score_size),
            np.hanning(self.score_size)
        )
        self.hann /= self.hann.sum()
    
    def init(self, image, bbox):
        """第一帧初始化"""
        x, y, w, h = bbox
        self.target_pos = np.array([x + w/2, y + h/2])
        self.target_sz = np.array([w, h])
        
        # 裁剪模板区域
        template = self._crop(image, self.target_pos, 
                             self.target_sz * (1 + self.context), 
                             (127, 127))
        self.template = torch.from_numpy(template).unsqueeze(0)
    
    def update(self, image):
        """更新目标位置"""
        # 裁剪搜索区域
        search_sz = self.target_sz * (1 + self.context) * 2
        search = self._crop(image, self.target_pos, search_sz, (255, 255))
        search = torch.from_numpy(search).unsqueeze(0)
        
        # 前向推理
        with torch.no_grad():
            response = self.model(self.template, search).squeeze().numpy()
        
        # 加入位置先验（抑制大幅度跳跃）
        response = response * (1 - 0.176) + 0.176 * self.hann
        
        # 找到峰值位置
        max_idx = np.unravel_index(response.argmax(), response.shape)
        dy, dx = max_idx[0] - self.score_size // 2, max_idx[1] - self.score_size // 2
        
        # 转换为图像坐标位移
        self.target_pos[0] += dx * self.stride
        self.target_pos[1] += dy * self.stride
        
        return (int(self.target_pos[0] - self.target_sz[0]/2),
                int(self.target_pos[1] - self.target_sz[1]/2),
                int(self.target_sz[0]), int(self.target_sz[1]))
    
    def _crop(self, image, pos, sz, output_sz):
        """裁剪并缩放到指定尺寸"""
        # 简化的图像裁剪逻辑
        pad = int(max(sz) // 2)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, 
                                     cv2.BORDER_REPLICATE)
        cx, cy = int(pos[0]) + pad, int(pos[1]) + pad
        half = int(max(sz) // 2)
        crop = padded[cy-half:cy+half, cx-half:cx+half]
        return cv2.resize(crop, output_sz).transpose(2, 0, 1).astype(np.float32)
```

## 深度学习关联

- **Siamese 跟踪的演进**：从 SiamFC (2016) 到 SiamRPN (2018) 引入区域提议机制、SiamMask (2019) 扩展到视频目标分割、SiamRPN++ (2019) 成功使用 ResNet 等深层网络、Ocean (2020) 引入目标感知特征聚合。Siamese 跟踪器的核心思想——离线训练的相似度匹配——已成为视觉跟踪的主流范式。
- **Transformer 跟踪器 (TransT, SwinTrack)**：将 Transformer 的交叉注意力机制引入目标跟踪，替代传统的互相关操作。模板特征和搜索特征通过自注意力和交叉注意力进行充分的信息交互，在遮挡、快速运动等挑战场景下显著超越基于互相关的 Siamese 方法。
- **多任务联合学习**：现代跟踪器趋向于将跟踪与分割、重检测、状态估计等多任务联合学习。如 MixFormer 将特征提取和目标搜索统一在 ViT 架构中迭代更新模板；OSTrack 提出统一的 Transformer 架构，取消模板分支和搜索分支的显式区分，让模型自适应地关注与目标相关的区域。
