# 53_OCR 光学字符识别与 CRNN 模型

## 核心概念

- **光学字符识别（Optical Character Recognition, OCR）**：从图像中识别出文字信息的技术，包括两个子任务——文本检测（定位文字在图像中的位置）和文本识别（将检测到的文字区域转录为字符串）。
- **CRNN（Convolutional Recurrent Neural Network）**：Shi et al. (2015) 提出的端到端文本识别模型，结合CNN（特征提取）、RNN（序列建模）和CTC（转录对齐）三部分。
- **CNN特征提取**：使用标准CNN（如VGG）从文本行图像中提取序列化的特征图——将特征图沿水平方向切分为特征序列，每个特征向量对应图像中的一个垂直条带。
- **双向LSTM序列建模**：使用双向LSTM对特征序列进行上下文建模，捕获字符之间的时序依赖关系（如"q"后面可能是"u"）。
- **CTC（Connectionist Temporal Classification）**：解决"输入序列长度与输出标签长度不对齐"问题的损失函数。CTC允许输出序列中有空白标记（blank），通过动态规划计算所有可能的对齐路径的概率和。
- **CTC的束搜索解码（Beam Search Decoding）**：推理时，使用带语言模型约束的束搜索（结合CTC路径概率和语言模型得分）找到最可能的转录结果。

## 数学推导

**CRNN的三大组件：**

- **CNN特征提取**：输入 $I \in \mathbb{R}^{H \times W \times 3}$ → 输出特征图 $F \in \mathbb{R}^{H' \times W' \times D}$
- **特征序列化**：按列划分 $X = [x_1, x_2, \dots, x_T]$，$T = W'$ 是序列长度
- **BiLSTM**：$h_t = \text{BiLSTM}(x_t, h_{t-1})$，输出 $\{y_t\}_{t=1}^T$

**CTC损失函数：**

给定输入序列 $x$（长度 $T$），标签序列 $l$（长度 $U \le T$），定义路径 $\pi = (\pi_1, \dots, \pi_T)$，其中 $\pi_t \in \mathcal{L} \cup \{blank\}$。

路径条件概率：
$$
p(\pi | x) = \prod_{t=1}^T y_{t, \pi_t}
$$

其中 $y_{t,k}$ 是时间步 $t$ 输出字符 $k$ 的概率。

标签序列的条件概率（所有路径的概率和）：
$$
p(l | x) = \sum_{\pi \in \mathcal{B}^{-1}(l)} p(\pi | x)
$$

其中 $\mathcal{B}$ 是"合并重复+删除blank"的映射函数。

**CTC损失（负对数似然）：**
$$
\mathcal{L}_{CTC} = -\log p(l | x)
$$

**CTC的前向后向算法（高效计算）：**
使用动态规划计算 $p(l | x)$，定义前向变量 $\alpha_t(s)$ 和 后向变量 $\beta_t(s)$，则：
$$
p(l | x) = \sum_{s} \alpha_t(s) \beta_t(s) / y_{t, l_s}
$$

## 直观理解

CRNN的工作方式可以类比为"看一行文字并读出它"。CNN部分负责"看"——从文本行图像中提取视觉特征（识别字符的笔画、轮廓等）；RNN部分负责"读"——按从左到右的顺序理解字符之间的关系（如"th"常一起出现）；CTC部分负责"对齐"——解决"看"和"读"之间的时间不一致问题。

CTC的核心问题可以这样理解：你有一张包含"hello"的图片，CNN产生了10个时间步的特征（每个时间步对应图片的一个垂直切片）。但"hello"只有5个字符，如何将10个特征帧映射到5个字符？CTC通过引入"blank"符号来插入不需要的帧，并合并重复的字符——比如"h_e_l_l_l_o"中的重复L被合并，blank被删除，最终得到"hello"。

## 代码示例

```python
import torch
import torch.nn as nn

class CRNN(nn.Module):
    """CRNN 文本识别模型 (简化版)"""
    def __init__(self, num_classes=37, hidden_size=256):  # 10 digits + 26 letters + blank
        super().__init__()
        # CNN特征提取 (简化)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4  (保持宽度方向)
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
            nn.Conv2d(512, 512, 2),  # (1, W/4-1)
        )
        
        # RNN序列建模
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True),
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W) 文本行图像
        x = self.cnn(x)  # (B, 512, 1, W')
        x = x.squeeze(2)  # (B, 512, W')
        x = x.permute(0, 2, 1)  # (B, W', 512) → 序列
        x, _ = self.rnn[0](x)
        x, _ = self.rnn[1](x)
        x = self.fc(x)  # (B, W', num_classes)
        return x

# CTC损失使用示例
def ctc_loss_example():
    # 模型输出 (B, T, num_classes)
    logits = torch.randn(2, 20, 37)
    log_probs = nn.functional.log_softmax(logits, dim=2)
    
    # 目标标签
    targets = torch.tensor([[3, 7, 7, 8, 14], [5, 4, 11, 11, 14]])  # "hello", "world"
    target_lengths = torch.tensor([5, 5])
    input_lengths = torch.tensor([20, 20])
    
    # CTC损失
    loss = nn.functional.ctc_loss(
        log_probs.permute(1, 0, 2),  # (T, B, C)
        targets, input_lengths, target_lengths, blank=36, reduction='mean'
    )
    return loss

# 测试
model = CRNN(num_classes=37)
x = torch.randn(1, 1, 32, 128)  # 标准文本行输入: 32x128
output = model(x)
print(f"CRNN输出: {output.shape}")  # (1, ~30, 37)

loss = ctc_loss_example()
print(f"CTC损失: {loss.item():.4f}")

# 解码 (简化: 贪婪解码)
def greedy_decode(output):
    """贪婪解码: 取每帧概率最高的字符, 合并重复, 去blank"""
    probs = nn.functional.softmax(output, dim=2)
    preds = probs.argmax(dim=2)  # (B, T)
    decoded = []
    for pred in preds:
        chars = []
        prev = -1
        for p in pred.tolist():
            if p != prev and p != 36:  # 36=blank
                chars.append(p)
            prev = p
        decoded.append(chars)
    return decoded

decoded = greedy_decode(output)
print(f"解码结果: {decoded}")
print(f"CRNN参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **端到端文本识别的标准**：CRNN + CTC成为场景文本识别的事实标准框架，被广泛应用在车牌识别、文档数字化、票据识别等场景。后续的改进包括使用注意力机制替代CTC（ASTER、SATRN）、使用Transformer编码器替代RNN（TRBA、ViTSTR）等。
- **文本检测+识别的完整OCR流水线**：工业OCR系统通常包含文本检测（如DBNet、EAST、PSENet）+ 文本识别（CRNN或其变体）。近年来的端到端方法（如End-to-End OCR、MASTER）试图将检测和识别统一到一个网络中。
- **从OCR到文档理解**：OCR技术正在从"文字识别"向"文档理解"演进——不仅要识别文字，还要理解文档结构、表格布局、逻辑关系，最终实现文档智能处理（Document AI）。
