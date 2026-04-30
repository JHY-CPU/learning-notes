# 28_3D 生成：Point-E 与 Shap-E

## 核心概念
- **Point-E (Point Cloud Diffusion)**：OpenAI 2022 年的 3D 生成模型，用扩散模型生成点云（Point Cloud）作为 3D 表示。先根据文本生成图像，再从图像生成 3D 点云。
- **Shap-E**：OpenAI 在同期的改进工作，直接生成 3D 隐式函数（如 NeRF 或 Signed Distance Function）的参数，而不是点云。生成的 3D 模型质量更高、更完整。
- **两阶段生成**：Point-E 采用"文本→图像→3D"的两阶段流水线。首先用文本到图像模型生成多视角图像，再用条件扩散模型从图像生成 3D 点云。
- **隐式 3D 表示**：Shap-E 不是直接生成点云或体素，而是生成一个神经辐射场（NeRF）或符号距离函数（SDF）的 MLP 权重——这是一种紧凑、高质量的 3D 表示。
- **参数化编码**：Shap-E 使用编码器将 3D 物体编码为 Transformer 的隐变量（latent），然后用扩散模型学习生成这些隐变量，最后用解码器渲染为 3D 网格或 NeRF。
- **效率优先**：Point-E 的设计优先考虑效率——30-60 秒即可生成一个 3D 模型，比之前的 3D 生成方法（DreamFusion 需要小时级）快几个数量级。

## 数学推导

**Point-E 的点云扩散**：

点云表示为 $P = \{p_1, p_2, ..., p_N\} \in \mathbb{R}^{N \times 3}$，$N$ 个点的坐标。

前向扩散（对点坐标加噪）：

$$
P_t = \sqrt{\bar{\alpha}_t} P_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

去噪网络以点云和条件图像 $I$ 为输入，预测噪声 $\epsilon_\theta(P_t, I, t)$。

**Shap-E 的两阶段流程**：

第一阶段（编码器-解码器训练）：

$$
\text{Encoder}: z = \mathcal{E}(V) \in \mathbb{R}^d, \quad \text{Decoder}: \hat{V} = \mathcal{D}(z)
$$

其中 $V$ 是 3D 物体的多视角渲染图像或 SDF 采样值，$z$ 是隐变量。

第二阶段（扩散模型训练）：

$$
L = \mathbb{E}_{z_0, c, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2 \right]
$$

其中 $c$ 是文本或图像条件。

**解码器输出**：

解码器输出一个 MLP $f_\theta: (x, y, z, d) \to (\text{RGB}, \sigma)$（NeRF 格式）或者 $f_\theta: (x, y, z) \to \text{SDF}$（隐式曲面格式）。

## 直观理解
- **Point-E = 先用文生图，再用图生 3D**：你输入"一把椅子"，系统先画一张椅子的图片，然后看着这张图片想象它的 3D 形状，生成一堆 3D 点。从不同角度看这些点，确实像一把椅子。
- **Shap-E = 直接生成 3D 模型的 DNA**：不再生成点云，而是生成一个能描述 3D 物体形状的"DNA"（隐变量）。这个 DNA 可以解码成一个函数，函数告诉你空间中的每个点是在物体内部还是外部，是什么颜色。
- **为什么需要两阶段**：文本到 3D 太难了——"一把舒适的扶手椅"这句话包含的信息不足以推断它的 3D 结构。先生成一张图像作为中间表示，相当于把问题分解为"文本→2D→3D"，每步更容易。
- **点云 vs 隐式函数**：点云像用笔在 3D 空间中标记一堆点（速度快但有空洞、不连续），隐式函数像用石膏雕刻（可以生成连续的表面、封闭的网格）。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class PointCloudDiffusion(nn.Module):
    """简化的点云扩散模型（类似 Point-E）"""
    def __init__(self, num_points=1024, image_emb_dim=512, hidden_dim=256):
        super().__init__()
        self.num_points = num_points
        
        # 图像条件编码
        self.image_encoder = nn.Sequential(
            nn.Linear(image_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 点云去噪 Transformer
        self.pos_embed = nn.Linear(3, hidden_dim)
        self.time_embed = nn.Embedding(1000, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=512,
                batch_first=True
            ),
            num_layers=6
        )
        
        self.output_proj = nn.Linear(hidden_dim, 3)
    
    def forward(self, point_cloud, image_emb, t):
        """
        point_cloud: [B, N, 3] 噪声点云
        image_emb: [B, D] 图像嵌入
        t: [B] 时间步
        """
        B, N, _ = point_cloud.shape
        
        # 嵌入
        point_feat = self.pos_embed(point_cloud)  # [B, N, D]
        time_feat = self.time_embed(t)[:, None, :].expand(-1, N, -1)
        
        # 图像条件（作为第一个 token 拼入序列）
        img_feat = self.image_encoder(image_emb)[:, None, :].expand(-1, N, -1)
        
        # 组合特征
        feat = point_feat + time_feat + img_feat
        feat = self.transformer(feat)
        
        # 预测噪声（用于更新点坐标）
        return self.output_proj(feat)  # [B, N, 3]

class ShapELatentDiffusion(nn.Module):
    """简化的 Shap-E 隐变量扩散模型"""
    def __init__(self, latent_dim=1024, text_dim=768):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器：从多视角图像提取 3D 隐变量
        self.encoder = nn.Sequential(
            nn.Linear(512 * 4, 2048),  # 4 个视角的 CLIP 嵌入
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )
        
        # 隐变量扩散
        self.diffusion = nn.Sequential(
            nn.Linear(latent_dim + text_dim + 256, 1024),  # 隐变量 + 文本 + 时间
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )
        
        self.time_embed = nn.Embedding(1000, 256)
    
    def encode_3d(self, images):
        """将多视角图像编码为 3D 隐变量"""
        return self.encoder(images)
    
    def decode_to_nerf(self, z, query_points):
        """
        从隐变量解码为 NeRF（简化）
        
        query_points: [B, M, 3] 查询空间坐标
        """
        # 实际 Shap-E 的输出是一个 MLP 权重
        # 此处简化为直接预测
        B, M, _ = query_points.shape
        z_expanded = z[:, None, :].expand(-1, M, -1)
        coords = query_points
        input_feat = torch.cat([z_expanded, coords], dim=-1)
        
        # 简化的 MLP 解码
        density = torch.sigmoid(torch.randn(B, M, 1))  # 密度
        color = torch.sigmoid(torch.randn(B, M, 3))  # 颜色
        return density, color

# Point-E 采样过程
def sample_point_cloud_text2image(point_model, text2image_model, text_prompt):
    """从文本生成点云（Point-E 两阶段）"""
    # 阶段 1: 文本 -> 图像
    with torch.no_grad():
        image = text2image_model(text_prompt)
        image_emb = torch.randn(1, 512)  # 使用 CLIP 编码图像
    
    # 阶段 2: 图像 -> 点云（扩散采样）
    point_model.eval()
    B = 1
    N = 1024
    pc_t = torch.randn(B, N, 3)
    
    for t in reversed(range(1000)):
        t_tensor = torch.full((B,), t, dtype=torch.long)
        noise_pred = point_model(pc_t, image_emb, t_tensor)
        
        alpha_bar = (t / 1000)  # 简化
        pc_t = (pc_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
    
    return pc_t

print("=== 3D 生成模型 ===")
print("\nPoint-E 流水线: 文本 -> 图像(CLIP) -> 点云扩散")
print("Shap-E 流水线: 文本/图像 -> 3D 隐变量 -> NeRF/SDF 解码")
print()
print("Point-E: 30-60 秒生成, 点云格式, 适合快速原型")
print("Shap-E: 几分钟生成, 网格/Nerf 格式, 适合高质量输出")

point_model = PointCloudDiffusion()
shap_model = ShapELatentDiffusion()
print(f"\nPoint-E 参数量: {sum(p.numel() for p in point_model.parameters()):,}")
print(f"Shap-E 参数量: {sum(p.numel() for p in shap_model.parameters()):,}")
```

## 深度学习关联
- **DreamFusion / Score Jacobian Chaining**：OpenAI 在 Point-E 同期的工作，通过 Score Distillation Sampling (SDS) 用预训练的文本到图像扩散模型指导 3D 生成（NeRF 优化），质量更高但需小时级计算。
- **Zero-1-to-3**：通过微调扩散模型学习"任意视角生成"——给定一张物体图，从任意新视角渲染该物体。这为 3D 生成提供了新的中间表示（多视角图→3D 重建）。
- **3D Gaussian Splatting**：2023 年的新 3D 表示方法，用数万个高斯椭球体表示 3D 场景，渲染速度远超 NeRF。DreamGaussian 等模型将其与扩散模型结合，实现了实时交互的 3D 生成。
- **MV-Diffusion / MVDream**：直接在扩散模型中生成多视角一致的图像，然后通过 3D 重建算法（如 NeuS）恢复 3D 模型，避免了 DreamFusion 的昂贵 SDS 优化过程。
