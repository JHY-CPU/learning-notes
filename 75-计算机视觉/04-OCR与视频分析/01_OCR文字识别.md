# OCR文字识别

*从CRNN到PaddleOCR — 光学字符识别技术全解析*

## 一、OCR 概述

**OCR（Optical Character Recognition，光学字符识别）**是将图像中的文字转换为可编辑文本的技术。现代OCR系统通常分为两个阶段：

输入图像 → 文字检测(Text Detection) → 文字识别(Text Recognition) → 输出文本

- **文字检测**：定位图像中文字区域的位置（边界框或多边形）
- **文字识别**：将裁剪出的文字图像转为文本序列

### OCR的挑战

| 挑战 | 说明 |
| --- | --- |
| 场景文字 | 自然场景中文字背景复杂、光照变化、透视变形 |
| 多语言 | 中英文混合、字符集庞大（中文常用字约3500+） |
| 艺术字体 | 变形、旋转、弯曲的文字 |
| 低质量 | 模糊、遮挡、低分辨率 |
| 版面复杂 | 表格、多栏、竖排文字 |

## 二、文字检测方法

### 基于分割的检测

- **PSENet**：从文字核心区域逐步扩展，分离相邻文本
- **DBNet（Differentiable Binarization）**：可微分二值化，端到端学习阈值

### 基于回归的检测

- **EAST**：直接回归文字框的四角坐标或旋转框
- **CRAFT**：检测文字的字符级区域和亲和力，擅长弯曲文字

## 三、CRNN 文字识别

CRNN（Convolutional Recurrent Neural Network, 2015）将CNN、RNN和CTC结合，是文字识别的经典架构：

文字图像 H×W×3 → CNN特征提取 → 序列化(按列切分) → BiLSTM → CTC解码 → 输出文本

```python
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)              # [B, 512, 1, T]
        conv = conv.squeeze(2)          # [B, 512, T]
        conv = conv.permute(0, 2, 1)    # [B, T, 512]
        rnn_out, _ = self.rnn(conv)     # [B, T, 512]
        output = self.fc(rnn_out)       # [B, T, num_classes]
        return output.permute(1, 0, 2)  # [T, B, num_classes] for CTC
```

## 四、CTC 对齐机制

文字识别面临**对齐问题**：输入序列长度T与输出文本长度不一致。

CTC引入**空白符（blank）**，允许网络在不同时刻输出相同字符：

- 合并连续重复字符：h-h-e-l-l-o → hello
- 移除空白符：h-blank-e-blank-l-l-o → hello

$$
L_CTC = -log P(y|x) = -log Σ_{π∈B⁻¹(y)} P(π|x)
$$

```python
ctc_loss = nn.CTCLoss(blank=0)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

## 五、Attention 与现代 OCR

### CTC vs Attention 对比

| 特性 | CTC | Attention |
| --- | --- | --- |
| 对齐方式 | 隐式（通过blank和合并） | 显式（注意力权重） |
| 字符依赖 | 假设条件独立 | 可建模依赖 |
| 解码速度 | 更快（贪心解码） | 较慢（自回归） |
| 长序列 | 更稳定 | 可能出现注意力漂移 |

### 现代方法

- **ABINet**：双向注意力语言模型增强视觉特征
- **TrOCR**：编码器ViT + 解码器RoBERTa，预训练OCR模型
- **PARSeq**：基于排列语言模型的场景文字识别

## 六、PaddleOCR

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr('image.jpg', cls=True)

for line in result[0]:
    bbox = line[0]          # 文字框坐标
    text = line[1][0]       # 识别文本
    confidence = line[1][1] # 置信度
    print(f'{text}: {confidence:.4f}')
```

## 七、Python 实战：OCR 模型训练

### 示例：CRNN + CTC 训练完整流程

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import string

class OCRDataset(Dataset):
    """OCR文字识别数据集"""

    def __init__(self, image_paths, labels, char_to_idx, max_len=25, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.image_paths[idx]).convert('L')  # 灰度
        if self.transform:
            img = self.transform(img)

        # 编码标签
        label = self.labels[idx]
        label_encoded = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        label_length = len(label_encoded)

        # Padding
        label_encoded = label_encoded + [0] * (self.max_len - label_length)
        return img, torch.tensor(label_encoded), torch.tensor(label_length)


def train_crnn_ocr():
    # 构建字符表
    chars = string.ascii_letters + string.digits + " .,-:;!?\"'"
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 保留给 blank
    num_classes = len(chars) + 1

    model = CRNN(num_classes=num_classes)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟数据
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    for epoch in range(50):
        model.train()
        # 模拟一个batch
        images = torch.randn(16, 1, 32, 128)  # [B, C, H, W]
        targets = torch.randint(1, num_classes, (16, 10))
        target_lengths = torch.full((16,), 10, dtype=torch.long)
        input_lengths = torch.full((16,), 32, dtype=torch.long)

        optimizer.zero_grad()
        output = model(images)  # [T, B, C]
        log_probs = output.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, CTC Loss: {loss.item():.4f}")

    return model
```

### 示例：DBNet 文字检测

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DBNet(nn.Module):
    """Differentiable Binarization Network for text detection"""

    def __init__(self, backbone_channels=[64, 128, 256, 512]):
        super().__init__()
        # 简化版特征金字塔
        self.conv1 = nn.Conv2d(backbone_channels[3], 64, 3, padding=1)
        self.conv2 = nn.Conv2d(backbone_channels[2], 64, 3, padding=1)
        self.conv3 = nn.Conv2d(backbone_channels[1], 64, 3, padding=1)

        # 预测头
        self.binarize = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.threshold = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def step_function(self, x, y):
        """可微分二值化"""
        return torch.reciprocal(1 + torch.exp(-50 * (x - y)))

    def forward(self, features):
        # features: 来自骨干网络的多尺度特征
        p = self.conv1(features[3])
        p = F.interpolate(p, size=features[2].shape[2:]) + self.conv2(features[2])
        p = F.interpolate(p, size=features[1].shape[2:]) + self.conv3(features[1])

        binary = self.binarize(p)
        threshold = self.threshold(p)

        # 可微分二值化
        approx_binary = self.step_function(binary, threshold)

        return binary, threshold, approx_binary
```

## 总结

- 现代OCR分为文字检测和文字识别两个阶段
- CRNN + CTC 是经典的文字识别方案，适合规则文本
- Attention 机制在弯曲、不规则文字上表现更好
- DBNet 的可微分二值化是文字检测的重要创新
- PaddleOCR 是目前最流行的开源OCR工具，支持多语言
- TrOCR 等基于Transformer的预训练模型代表了OCR的最新方向


<!-- Converted from: 01_OCR文字识别.html -->
