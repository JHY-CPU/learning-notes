# 41_Deepfake 检测技术与防御

## 核心概念

- **Deepfake（深度伪造）**：利用深度学习（尤其是生成模型如 GAN、扩散模型）创建的逼真假图像、视频和音频，常被用于虚假信息传播、身份冒充和敲诈。
- **检测技术分类**：Deepfake 检测方法分为——(1) 基于图像伪影的检测（如频率域分析、面部关键点异常），(2) 基于深度学习分类器的检测，(3) 基于生成溯源的水印检测。
- **频率域差异 (Frequency Domain Analysis)**：GAN/扩散模型生成的图像在频率域有特定模式——高频分量不足（过度平滑）或特定频率峰值，可通过 FFT/DCT 变换检测。
- **生物信号不一致**：Deepfake 视频中的眨眼频率异常、心跳/脉搏信号缺失（远程光电容积描记术 rPPG）、呼吸模式异常等生物信号的物理不一致性。
- **对抗性检测**：检测器和生成器之间形成"军备竞赛"——生成器学会绕过检测器的弱点，检测器需要不断更新适应新类型的 Deepfake。
- **后门攻击 (Backdoor Attack)**：检测器本身可能被攻击者植入后门——特定输入模式会触发错误分类。
- **溯源技术 (Forensic Trace)**：训练数据中固有的相机噪声模型（Photo-Response Non-Uniformity, PRNU）在生成图像中消失，可作为检测线索。

## 数学推导

**频率域特征提取**：

对图像 $x$ 做二维离散傅里叶变换（DFT）：

$$
F(u, v) = \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x(h, w) \cdot e^{-j2\pi(\frac{uh}{H} + \frac{vw}{W})}
$$

功率谱：$P(u, v) = |F(u, v)|^2$

真实图像和生成图像的功率谱在高低频分布上有系统差异。

**DCT 系数的统计分布**：

对图像块进行 DCT 变换后，DCT 系数的分布服从广义高斯分布：

$$
p(z; \alpha, \beta) = \frac{\beta}{2\alpha\Gamma(1/\beta)} \exp\left(-\left|\frac{z}{\alpha}\right|^\beta\right)
$$

形状参数 $\beta$ 在真实和生成图像中有统计差异。

**深度检测分类器**：

$$
\hat{y} = \sigma(f_\theta(\phi(x)))
$$

其中 $\phi(x)$ 是特征提取器（如 EfficientNet、XceptionNet），$f_\theta$ 是分类头，$\sigma$ 是 Sigmoid 激活。训练损失为二值交叉熵：

$$
\mathcal{L} = -y \log\hat{y} - (1-y)\log(1-\hat{y})
$$

**通用检测对抗训练**：

为了提升检测器的鲁棒性，用对抗样本训练：

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y))
$$

检测器在原始和对抗样本上同时训练。

**水印技术 (Stable Signature)**：

将固定密钥 $k$ 编码到生成图像中的不可见模式：

$$
x_{\text{watermarked}} = x + \delta_\theta(x, k), \quad \|\delta_\theta\|_\infty < \epsilon
$$

解码器可从带水印图像中恢复密钥 $k$，即使经过裁剪、压缩等变换。

## 直观理解

- **频率域检测 = 数字图像的"笔迹"分析**：就像书法家看笔迹判断真伪，频率域检测分析"图像的笔迹"——真实照片有自然的高频细节（如头发丝、皮肤纹理），AI 生成的图像在这些细节上有系统性的"不自然"。
- **生物信号检测 = 看出视频中的人"不像活人"**：Deepfake 视频可能看起来逼真，但人眼不易察觉的生理信号（如微表情、瞳孔反射、脉搏导致的面部微色变）会暴露"这不是真人"。
- **检测-生成的军备竞赛**：这就像防病毒软件和病毒之间的博弈——生成器不断改进以逃避检测，检测器不断更新以发现新的伪造痕迹。这是一个没有终点的猫鼠游戏。
- **水印机制 = AI 生成内容的"钢印"**：就像人民币的水印防伪，在生成时嵌入不可见的数字水印，无论经过多少次转发、压缩、截屏，水印仍然可检测。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyDomainAnalyzer:
    """频率域分析器：检测 Deepfake 的频率伪影"""
    @staticmethod
    def compute_fft_spectrum(image):
        """计算图像的 FFT 功率谱"""
        # image: [B, C, H, W]
        fft = torch.fft.fft2(image, norm='ortho')
        fft_shift = torch.fft.fftshift(fft)
        power_spec = torch.abs(fft_shift) ** 2
        return power_spec
    
    @staticmethod
    def compute_radial_profile(power_spec):
        """计算径向功率谱"""
        B, C, H, W = power_spec.shape
        center_h, center_w = H // 2, W // 2
        
        max_radius = min(center_h, center_w)
        radial_profile = torch.zeros(B, C, max_radius)
        
        for r in range(max_radius):
            # 创建半径为 r 的环形掩码
            y, x = torch.meshgrid(
                torch.arange(H), torch.arange(W), indexing='ij'
            )
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            mask = (dist >= r) & (dist < r + 1)
            
            for b in range(B):
                for c in range(C):
                    vals = power_spec[b, c][mask]
                    radial_profile[b, c, r] = vals.mean() if vals.numel() > 0 else 0
        
        return radial_profile  # 低频→高频的能量分布
    
    @staticmethod
    def extract_high_freq_ratio(power_spec, threshold_ratio=0.1):
        """
        计算高频能量占总能量的比例
        生成图像通常高频能量低于真实图像
        """
        B, C, H, W = power_spec.shape
        center_h, center_w = H // 2, W // 2
        noise_radius = int(min(H, W) * threshold_ratio)
        
        # 总能量
        total_energy = power_spec.sum(dim=(2, 3))
        
        # 高频能量（消除低频中心区域）
        mask = torch.ones(H, W, device=power_spec.device)
        mask[center_h-noise_radius:center_h+noise_radius,
             center_w-noise_radius:center_w+noise_radius] = 0
        high_freq_energy = (power_spec * mask[None, None, :, :]).sum(dim=(2, 3))
        
        return high_freq_energy / total_energy

class DeepfakeDetector(nn.Module):
    """
    Deepfake 检测分类器
    基于 EfficientNet 特征提取 + 频率域特征的融合
    """
    def __init__(self):
        super().__init__()
        # 空间域特征提取
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 频率域特征提取（从 FFT 幅值）
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # 空间特征
        spatial_feat = self.spatial_encoder(x).flatten(1)
        
        # 频率特征
        fft_spec = torch.fft.fft2(x, norm='ortho')
        fft_abs = torch.abs(torch.fft.fftshift(fft_spec))
        freq_feat = self.freq_encoder(fft_abs).flatten(1)
        
        # 融合
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(combined)

# 对抗训练
def adversarial_training_step(detector, real_images, fake_images, epsilon=0.01):
    """使用对抗训练增强检测器鲁棒性"""
    # 对真实图像添加对抗扰动
    real_images.requires_grad_(True)
    pred_real = detector(real_images)
    loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
    
    grad = torch.autograd.grad(loss_real, real_images, create_graph=False)[0]
    adv_real = real_images + epsilon * grad.sign()
    adv_real = adv_real.detach()
    
    # 在原始和对抗样本上训练
    all_images = torch.cat([
        real_images.detach(), adv_real, fake_images.detach()
    ], dim=0)
    labels = torch.cat([
        torch.ones(real_images.size(0)),
        torch.ones(real_images.size(0)),
        torch.zeros(fake_images.size(0)),
    ])
    
    pred = detector(all_images)
    loss = F.binary_cross_entropy(pred, labels[:, None])
    
    return loss

print("=== Deepfake 检测技术与防御 ===")
detector = DeepfakeDetector()
real = torch.randn(4, 3, 224, 224)
fake = torch.randn(4, 3, 224, 224)

# 频率分析
freq_ratio_real = FrequencyDomainAnalyzer.extract_high_freq_ratio(
    FrequencyDomainAnalyzer.compute_fft_spectrum(real)
)
print(f"真实图像高频能量比: {freq_ratio_real.mean().item():.4f}")

freq_ratio_fake = FrequencyDomainAnalyzer.extract_high_freq_ratio(
    FrequencyDomainAnalyzer.compute_fft_spectrum(fake)
)
print(f"生成图像高频能量比: {freq_ratio_fake.mean().item():.4f}")

# 检测
pred = detector(real)
print(f"检测器输出: {pred.mean().item():.3f} (应接近 1)")
print(f"检测器参数量: {sum(p.numel() for p in detector.parameters()):,}")
```

## 深度学习关联

- **Diffusion 时代的新深度伪造**：Stable Diffusion 生成的图像比 GAN 更逼真、更少伪影，传统检测方法的准确率下降。扩散模型的检测需要新的特征：如毛孔纹理的统计异常、CLIP 特征空间中的异常。
- **深度伪造检测的"泛化"挑战**：检测器在一个类型（如 StyleGAN 生成）上训练，在另一个类型（如扩散模型生成）上的检测准确率显著下降。领域泛化和跨域检测是研究热点。
- **防御性水印 (Glaze, Nightshade)**：艺术家使用 Glaze 等工具在发布作品前添加人类不可见但 AI 可感知的扰动，使模型无法从该作品中学习到有用的风格特征——这是一种主动防御。
- **法规与技术并重**：C2PA（内容来源和真实性联盟）的标准为数字内容提供可验证的来源元数据；美国的《DEEPFAKES Accountability Act》、欧盟的《AI Act》等法规为法律追责提供框架。
