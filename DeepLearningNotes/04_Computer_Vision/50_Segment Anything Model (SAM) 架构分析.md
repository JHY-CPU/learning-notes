# 50_Segment Anything Model (SAM) 架构分析

## 核心概念

- **Segment Anything Model (SAM)**：Meta AI (2023) 提出的"提示分割"（promptable segmentation）基础模型，可以基于点、框、文本等提示对任意图像中的任意物体进行分割。
- **SAM的三大组件**：**(1)** 图像编码器（Image Encoder）——使用MAE预训练的ViT-Huge提取图像特征；**(2)** 提示编码器（Prompt Encoder）——编码点、框、掩码等提示信息；**(3)** 掩码解码器（Mask Decoder）——融合图像和提示特征，输出分割掩码。
- **SA-1B数据集**：与SAM一同发布的超大分割数据集，包含11M张图像和1.1B个掩码，使用SAM+人工标注的"数据引擎"循环生成。
- **提示分割任务**：给定一个提示（如一个点或一个框），SAM输出该提示所指物体的分割掩码。模型被设计为"提示无关"——不管提示是什么，都能产生合理的分割。
- **歧义感知输出**：当一个提示存在歧义时（如"一个点"可能指代多个物体），SAM可输出多个有效掩码（通过预测的IoU得分排序），而不是只输出一个。
- **零样本泛化**：SAM在未见过的图像、物体类别和任务上展现出强大的零样本分割能力，无需微调即可用于新场景。

## 数学推导

**图像编码器：**
使用MAE预训练的ViT-Huge：
$$
F = \text{ViT-H}(I) \in \mathbb{R}^{H/16 \times W/16 \times D}
$$

其中 $D=256$ 是输出特征维度。

**提示编码器：**

- 点提示（稀疏）：每个点编码为位置编码 + 类型（前景/背景）嵌入的和
- 框提示（稀疏）：编码框的角点位置编码 + 可学习嵌入表示"框"
- 掩码提示（密集）：使用卷积下采样编码
- 文本提示（稀疏）：使用CLIP文本编码器

**掩码解码器（修改的Transformer解码器）：**

解码器接收两类Token：
- 图像Token：从图像特征采样的token
- 输出Token：可学习的Token，负责生成掩码

解码器结构：
1. 双向注意力（Token之间）
2. Token到图像注意力（Token attend 图像特征）
3. 图像到Token注意力（图像特征 attend Token）
4. MLP

输出：每个Token预测一个 $H \times W$ 的掩码（通过点积计算）。

**SAM的损失函数：**
$$
\mathcal{L} = \lambda_{dice} \mathcal{L}_{dice} + \lambda_{bce} \mathcal{L}_{bce}
$$

其中 $\mathcal{L}_{dice}$ 是Dice损失（处理类别不平衡），$\mathcal{L}_{bce}$ 是二值交叉熵损失。

**IoU预测头：**
SAM还预测每个掩码的IoU分数（解码器的一个额外输出分支），用于在推理时对多个候选掩码进行排序和选择。

## 直观理解

SAM的工作方式类似于"视觉版的聊天机器人"——你给出"提示"（如用鼠标点击或画框），SAM根据提示给出相应的分割结果。你可以通过改进提示来迭代优化结果（"不对，不是这只猫，是旁边那只"）。

这种"提示分割"模式将三个传统独立的任务（交互式分割、自动分割、开放世界分割）统一到了一个框架中。图像编码器像是一个"场景理解引擎"——一次性提取图像中所有物体的通用特征；然后结合提示编码器的指令（你想分割什么），掩码解码器快速产生分割结果。

MAE预训练使SAM的图像编码器对物体的边界和形状有深刻的理解，不需要针对特定类别进行训练就能分割几乎所有物体。

## 代码示例

```python
import torch
import torch.nn as nn

class SAMMaskDecoder(nn.Module):
    """SAM 掩码解码器 (简化版)"""
    def __init__(self, embed_dim=256, num_output_masks=3):
        super().__init__()
        self.num_output_masks = num_output_masks
        
        # 输出Token (负责生成掩码)
        self.output_tokens = nn.Embedding(num_output_masks, embed_dim)
        # IoU Token (预测掩码质量)
        self.iou_token = nn.Embedding(1, embed_dim)
        
        # 简化的Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 掩码预测头 (通过点积与图像特征计算)
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # IoU预测头
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, image_embeddings, prompt_tokens):
        """
        image_embeddings: (B, N_img_tokens, D) 图像特征
        prompt_tokens: (B, N_prompt_tokens, D) 提示编码
        """
        B = image_embeddings.shape[0]
        
        # 拼接输出Token和IoU Token
        output_tokens = self.output_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        iou_token = self.iou_token.weight.unsqueeze(0).expand(B, -1, -1)
        query_tokens = torch.cat([iou_token, output_tokens, prompt_tokens], dim=1)
        
        # Transformer解码 (cross-attend to image features)
        decoded = self.decoder(query_tokens, image_embeddings)
        
        iou_token_out = decoded[:, 0, :]
        output_tokens_out = decoded[:, 1:1+self.num_output_masks, :]
        
        # 预测掩码
        mask_embeddings = self.mask_head(output_tokens_out)
        masks = torch.einsum('bnd,bmd->bnm', mask_embeddings, image_embeddings)
        
        # 预测IoU
        iou_pred = self.iou_head(iou_token_out)
        
        return masks.view(B, self.num_output_masks, -1), iou_pred

class SAM(nn.Module):
    """简化的 SAM 模型"""
    def __init__(self):
        super().__init__()
        # 图像编码器 (简化: 用线性层代替ViT)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((16, 16)),  # 固定尺寸输出
        )
        self.embed_proj = nn.Linear(256, 256)
        
        # 掩码解码器
        self.mask_decoder = SAMMaskDecoder(embed_dim=256)
        
        # 提示编码 (简化的点/框编码)
        self.point_embed = nn.Embedding(2, 256)  # 前景/背景

    def forward(self, image, point_coords, point_labels):
        # 图像编码
        img_feat = self.image_encoder(image)  # (B, 256, 16, 16)
        img_feat = img_feat.flatten(2).transpose(1, 2)
        img_feat = self.embed_proj(img_feat)  # (B, 256, 256)
        
        # 提示编码 (简化: 直接位置编码)
        prompt_tokens = self.point_embed(point_labels.long())
        
        # 解码
        masks, iou_pred = self.mask_decoder(img_feat, prompt_tokens)
        return masks, iou_pred

# 测试
model = SAM()
img = torch.randn(1, 3, 256, 256)
# 模拟点击提示: 2个点 (前景)
point_coords = torch.tensor([[[100, 100], [150, 150]]], dtype=torch.float32)
point_labels = torch.zeros(1, 2, dtype=torch.long)  # 0=前景

masks, iou_pred = model(img, point_coords, point_labels)
print(f"预测掩码数: {masks.shape[1]}")
print(f"IoU预测: {iou_pred}")

print(f"\nSAM参数量 (简化版): {sum(p.numel() for p in model.parameters()):,}")
print("\nSAM的关键特性:")
print("- 提示分割: 点/框/掩码/文本 → 分割掩码")
print("- 歧义感知: 一个提示可输出多个有效掩码")
print("- 零样本: 无需针对新场景重新训练")
print("- 实时交互: 图像编码一次，可交互分割多次")
```

## 深度学习关联

- **计算机视觉的"基础模型"**：SAM是计算机视觉领域第一个真正意义上的"基础模型"（Foundation Model）之一，类似于NLP中的GPT——一个模型解决大量分割任务（交互式分割、边缘检测、物体提议生成、分割一切等）。
- **视觉分割的范式变革**：SAM将分割从"为特定任务训练特定模型"转变为"一个模型做所有分割"。这极大简化了分割在具体应用中的部署流程——不需要收集标注数据、不需要训练、直接使用SAM即可。
- **SAM生态的扩展**：SAM催生了庞大的扩展生态——SAM-Adapter（适配下游任务）、SAM-Med2D（医学图像）、SAM-Track（视频跟踪）、MobileSAM（高效版SAM，用于移动端）、LangSAM（结合Grounding DINO实现文本驱动的分割一切）等。
