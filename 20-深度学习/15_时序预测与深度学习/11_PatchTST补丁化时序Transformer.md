# 11_PatchTST：补丁化时序 Transformer

## 1. 核心思想

PatchTST (ICLR 2023) 借鉴 Vision Transformer (ViT) 的 patch 思想，将长时间序列分割为**子序列补丁 (Patch)**，每个补丁作为一个 token 输入 Transformer。

**关键创新：**
1. **补丁化 (Patching)**：将长序列切分为固定长度的子序列，大幅减少 token 数
2. **通道独立性 (Channel Independence)**：每个变量独立处理，类比 ViT 的通道处理方式

```
原始序列 (L=512, D=7):
[x₁, x₂, x₃, ..., x₅₁₂] × 7 个变量

补丁化 (patch_len=16, stride=8):
[x₁...x₁₆] [x₉...x₂₄] [x₁₇...x₃₂] ... → 63 个 token
```

## 2. 补丁嵌入

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将时间序列分割为补丁并嵌入"""
    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 线性投影：每个 patch 映射到 d_model 维
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, 1) 单个变量的序列
        B, L, C = x.shape
        x = x.transpose(1, 2)  # (B, 1, L)

        # 滑动窗口提取 patch
        # unfold: (B, 1, num_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: (B, C, num_patches, patch_len)
        num_patches = patches.size(2)
        patches = patches.reshape(B * C, num_patches, self.patch_len)

        # 线性投影
        embeddings = self.projection(patches)  # (B*C, num_patches, d_model)
        embeddings = self.dropout(embeddings)

        return embeddings, num_patches

# 示例
x = torch.randn(32, 512, 1)  # 单变量
patch_emb = PatchEmbedding(patch_len=16, stride=8, d_model=128)
emb, n_patches = patch_emb(x)
print(f'补丁数: {n_patches}, 嵌入维度: {emb.shape}')
# 补丁数: 63, 嵌入维度: torch.Size([32, 63, 128])
```

## 3. 通道独立性

**核心思想：** 将多变量时序的每个变量视为独立样本，共享同一个 Transformer 参数。

```
多变量输入 (B, L, D):
  变量1: [x₁, x₂, ..., x_L] → 补丁化 → Transformer → 预测
  变量2: [x₁, x₂, ..., x_L] → 补丁化 → Transformer → 预测  (共享参数)
  ...
  变量D: [x₁, x₂, ..., x_L] → 补丁化 → Transformer → 预测
```

**优势：**
- 参数量与变量数无关，可扩展到高维时序
- 避免变量间的噪声干扰
- 隐式正则化效果

```python
class ChannelIndependentPatchTST(nn.Module):
    def __init__(self, n_vars, patch_len, stride, d_model,
                 n_heads, n_layers, d_ff, pred_len, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len

        # 共享的单变量模型
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)

        # 位置编码
        self.max_patches = 200
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_patches, d_model) * 0.02
        )

        # Transformer 编码器（所有变量共享）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # 预测头
        self.head = nn.Linear(d_model, patch_len)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # 转换为每个变量独立处理: (B*D, L, 1)
        x = x.permute(0, 2, 1).reshape(B * D, L, 1)

        # 补丁嵌入
        patches, num_patches = self.patch_embedding(x)  # (B*D, N, d_model)

        # 添加位置编码
        patches = patches + self.pos_encoding[:, :num_patches, :]

        # Transformer 编码
        encoded = self.encoder(patches)  # (B*D, N, d_model)

        # 预测：每个 patch 预测 patch_len 个未来值
        # 然后拼接并截取 pred_len
        pred_patches = self.head(encoded)  # (B*D, N, patch_len)

        # 重叠拼接（stride < patch_len 时需要平均）
        pred = self._merge_patches(pred_patches, L, num_patches)
        pred = pred.reshape(B, D, self.pred_len).permute(0, 2, 1)

        return pred

    def _merge_patches(self, patches, seq_len, num_patches):
        """将重叠的预测补丁合并为完整预测序列"""
        BxD, N, P = patches.shape
        output = torch.zeros(BxD, self.pred_len, device=patches.device)
        count = torch.zeros(BxD, self.pred_len, device=patches.device)

        stride = seq_len // N  # 近似 stride
        for i in range(N):
            start = i * stride
            end = min(start + P, self.pred_len)
            output[:, start:end] += patches[i, :, :end-start]
            count[:, start:end] += 1

        return output / count.clamp(min=1)
```

## 4. 完整 PatchTST

```python
class PatchTST(nn.Module):
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16,
                 stride=8, d_model=128, n_heads=8, n_layers=3,
                 d_ff=256, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        num_patches = (seq_len - patch_len) // stride + 1

        # 补丁嵌入
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # 可选：回归标记（用于聚合全局信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                        dropout, batch_first=True, activation='gelu'),
            n_layers
        )

        # 预测头
        self.head = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        # 通道独立：转为 (B*D, L, 1)
        x = x.permute(0, 2, 1).reshape(B * D, L, 1)
        x = x.squeeze(-1)  # (B*D, L)

        # 提取补丁
        patches = x.unfold(1, self.patch_len, self.stride)  # (B*D, N, P)
        N = patches.size(1)
        patches = self.patch_proj(patches) + self.pos_embed[:, :N, :]

        # 添加 CLS token
        cls = self.cls_token.expand(B * D, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)  # (B*D, N+1, d_model)

        # 编码
        encoded = self.encoder(tokens)

        # 取 CLS token 的输出进行预测
        cls_output = encoded[:, 0, :]  # (B*D, d_model)
        pred = self.head(cls_output)   # (B*D, pred_len)

        # 恢复形状
        pred = pred.reshape(B, D, self.pred_len).permute(0, 2, 1)
        return pred
```

## 5. 补丁参数选择

| 序列长度 L | patch_len | stride | token 数 | 效果 |
|-----------|-----------|--------|---------|------|
| 96        | 8         | 4      | 23      | 短序列适用 |
| 96        | 16        | 8      | 11      | 推荐 |
| 336       | 16        | 8      | 41      | 推荐 |
| 720       | 24        | 12     | 59      | 长序列适用 |
| 1440      | 32        | 16     | 89      | 超长序列 |

**经验法则：**
- patch_len 应包含一个完整的基本模式周期
- stride 通常取 patch_len 的一半（50% 重叠）
- token 数控制在 20-100 之间较为合理

## 6. PatchTST vs 其他模型

| 特性 | PatchTST | Informer | Autoformer |
|------|----------|----------|------------|
| Token 化方式 | 补丁（子序列） | 时间步 | 时间步 |
| Token 数 | $O(L/p)$ | $O(L)$ | $O(L)$ |
| 注意力复杂度 | $O((L/p)^2)$ | $O(L \log L)$ | $O(L \log L)$ |
| 通道建模 | 独立 | 混合 | 混合 |
| 代码简洁性 | 非常简洁 | 中等 | 复杂 |

## 7. 实战使用

```python
model = PatchTST(
    n_vars=7,
    seq_len=512,
    pred_len=96,
    patch_len=16,
    stride=8,
    d_model=128,
    n_heads=8,
    n_layers=3,
    d_ff=256
)

x = torch.randn(32, 512, 7)
pred = model(x)  # (32, 96, 7)
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
```

---

**要点总结：**
- PatchTST 通过补丁化将长序列压缩为少量 token，大幅降低计算量
- 通道独立性是其成功的关键：简化建模、提升泛化
- 补丁设计借鉴 ViT，但针对时序数据特点进行了适配
- 架构简洁、效果强劲，是目前最值得推荐的时序 Transformer 基线之一
