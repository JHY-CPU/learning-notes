# OCR文字识别

*从CRNN到PaddleOCR — 光学字符识别技术全解析*


**OCR（Optical Character Recognition，光学字符识别）**是将图像中的文字转换为可编辑文本的技术。现代OCR系统通常分为两个阶段：

输入图像
→
文字检测
(Text Detection)
→
文字识别
(Text Recognition)
→
输出文本

- **文字检测**
   ：定位图像中文字区域的位置（边界框或多边形）
- **文字识别**
   ：将裁剪出的文字图像转为文本序列
- 端到端方法可同时完成检测和识别


### OCR的挑战


| 挑战 | 说明 |
| --- | --- |
| 场景文字 | 自然场景中文字背景复杂、光照变化、透视变形 |
| 多语言 | 中英文混合、字符集庞大（中文常用字约3500+） |
| 艺术字体 | 变形、旋转、弯曲的文字 |
| 低质量 | 模糊、遮挡、低分辨率 |
| 版面复杂 | 表格、多栏、竖排文字 |


### 2.1 基于分割的检测


将文字检测视为像素级分割问题，预测每个像素是否属于文字区域：


- **语义分割**
   ：使用FCN、U-Net等网络输出文字/非文字mask
- **PSENet（Progressive Scale Expansion Network）**
   ：从文字核心区域逐步扩展，分离相邻文本
- **DBNet（Differentiable Binarization）**
   ：可微分二值化，端到端学习阈值


> **Note:** **DBNet核心创新：**
> 传统方法使用固定阈值将概率图二值化，DBNet将二值化过程变为可微分的，让网络自适应学习最佳阈值，显著提升检测精度和速度。


### 2.2 基于回归的检测


- **EAST（Efficient and Accurate Scene Text detector）**
   ：直接回归文字框的四角坐标或旋转框
- **TextBoxes**
   ：修改SSD的anchor为长条形，适配文字特点
- **CRAFT**
   ：检测文字的字符级区域和亲和力，擅长弯曲文字


### 3.1 架构


CRNN（Convolutional Recurrent Neural Network, 2015）将CNN、RNN和CTC结合，是文字识别的经典架构：

文字图像
H×W×3
→
CNN特征提取
(VGG/ResNet)
→
序列化
(按列切分)
→
BiLSTM
序列建模
→
CTC解码
输出文本

### 3.2 各组件详解


**CNN特征提取器：**


- 通常使用VGG-like或ResNet浅层网络
- 将高度H压缩到1，宽度W保留为序列长度
- 输出特征图: 1×T×C（T为时间步数，C为特征维度）


**双向LSTM：**


- 捕获上下文依赖关系
- 2层BiLSTM，隐藏层维度256
- 输出: T×V（V为字符表大小）


```
# CRNN PyTorch实现
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        # RNN部分
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        # 输出层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)              # [B, 512, 1, T]
        conv = conv.squeeze(2)          # [B, 512, T]
        conv = conv.permute(0, 2, 1)    # [B, T, 512]
        rnn_out, _ = self.rnn(conv)     # [B, T, 512]
        output = self.fc(rnn_out)       # [B, T, num_classes]
        return output.permute(1, 0, 2)  # [T, B, num_classes] for CTC
```


### 4.1 问题


文字识别面临**对齐问题**：输入序列长度T（特征图宽度）与输出文本长度不一致，且没有明确的字符级对齐标注。


### 4.2 CTC（Connectionist Temporal Classification）


CTC引入**空白符（blank）**，允许网络在不同时刻输出相同字符，并通过合并规则将输出序列映射为最终文本：


- 合并连续重复字符：hh-e-l-l-o → hello
- 移除空白符：h-blank-e-blank-l-l-o → hello
- 路径概率 = 所有能映射到目标文本的路径概率之和


$$
L_CTC = -log P(y|x) = -log Σ_{π∈B⁻¹(y)} P(π|x)
$$


```
# CTC解码示例
# 路径: "h--ee-ll-lo" → 合并重复 → "heello" → 去blank → "hello"
# 前向-后向算法高效计算所有路径的概率和

# PyTorch CTC Loss
ctc_loss = nn.CTCLoss(blank=0)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
# log_probs: [T, B, C] (需要log_softmax)
# targets: [B, S] (目标序列)
# input_lengths: [B] (每个样本的输入长度)
# target_lengths: [B] (每个样本的目标长度)
```


### 5.1 注意力机制在OCR中的应用


CTC假设输出之间条件独立，无法建模字符间的依赖关系。**Attention机制**允许解码器在每一步动态关注输入的不同位置：


```
# Encoder-Decoder with Attention
编码器: CNN → 特征序列 [T, D]
解码器: 每一步 t:
  1. 计算注意力权重: α_t = Attention(h_{t-1}, 特征序列)
  2. 上下文向量: c_t = Σ α_t · 特征序列
  3. 预测字符: y_t = Decoder(h_{t-1}, y_{t-1}, c_t)
```


### 5.2 SAR（Show, Attend and Read）


- 2D注意力机制，在特征图的空间位置上计算注意力
- 适合不规则文字（弯曲、旋转）
- 解码器使用LSTM + Attention


### 5.3 CTC vs Attention对比


| 特性 | CTC | Attention |
| --- | --- | --- |
| 对齐方式 | 隐式（通过blank和合并） | 显式（注意力权重） |
| 字符依赖 | 假设条件独立 | 可建模依赖 |
| 解码速度 | 更快（贪心解码） | 较慢（自回归） |
| 长序列 | 更稳定 | 可能出现注意力漂移 |
| 精度 | 中等 | 通常更高 |


### 6.1 ABINet


使用双向注意力语言模型增强视觉特征：


- 视觉模型提取字符特征
- 语言模型（Transformer编码器）建模字符间关系
- 迭代精化：视觉→语言→视觉→...


### 6.2 TrOCR


微软提出的预训练OCR模型：


- 编码器：预训练的ViT/BEiT（图像理解）
- 解码器：预训练的RoBERTa/DeBERTa（语言生成）
- 大规模合成数据预训练 + 真实数据微调


### 6.3 PARSeq


基于Permutation Language Modeling的场景文字识别：


- 训练时随机排列字符顺序，增强泛化能力
- 支持自回归和非自回归解码
- 处理不规则文字效果好


### 7.1 系统架构


PaddleOCR是百度开源的实用OCR工具套件，包含完整的检测、方向分类、识别三阶段流程：

输入图像
→
DB检测
(PP-OCR)
→
方向分类
(文本旋转)
→
SVTR识别
(CRNN)
→
后处理

### 7.2 PP-OCR系列


| 版本 | 检测模型 | 识别模型 | 特点 |
| --- | --- | --- | --- |
| PP-OCRv2 | DB + LKPAN | SVTR-LCNet | 轻量化，移动端友好 |
| PP-OCRv3 | DB + LKPAN | SVTR-LCNet v2 | GTC策略，精度提升 |
| PP-OCRv4 | DB + LKPAN | SVTRv2 | 更高精度，多语言支持 |


```
# PaddleOCR使用示例
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr('image.jpg', cls=True)

for line in result[0]:
    bbox = line[0]          # 文字框坐标 [[x1,y1], [x2,y2], ...]
    text = line[1][0]       # 识别文本
    confidence = line[1][1] # 置信度
    print(f'{text}: {confidence:.4f}')
```


### 8.1 弯曲文字检测


自然场景中的文字经常呈弯曲或不规则形状：


- **基于分割**
   ：预测文字区域mask后通过最小外接矩形或多边形拟合
- **基于回归**
   ：直接预测文字边界的多个控制点（如Bezier曲线、多边形顶点）
- **ABCNet**
   ：使用Bezier曲线参数化文字边界，自适应Bezier RoI Align提取特征


### 8.2 端到端检测识别


- **FOTS**
   ：共享特征的检测和识别，RoIRotate对齐文字区域
- **Mask TextSpotter**
   ：实例分割 + 字符级识别
- **ABCNet v2**
   ：Bezier曲线检测 + 自适应特征提取 + 识别


<!-- Converted from: 01_OCR文字识别.html -->
