# 6_隐式神经表示 (INR)

## 1. 什么是隐式神经表示

**隐式神经表示 (Implicit Neural Representations, INR)** 用一个神经网络来连续地表示信号（图像、音频、3D形状等），输入坐标，输出该位置的信号值。

$$f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^c$$

- 对于图像：$(x, y) \rightarrow (r, g, b)$
- 对于3D形状：$(x, y, z) \rightarrow s$（SDF或占用概率）
- 对于音频：$t \rightarrow$ 振幅

### 1.1 与显式表示的对比

| 特性 | 显式表示 (像素/体素) | 隐式表示 (INR) |
|------|---------------------|----------------|
| 分辨率 | 固定 | 连续/无限 |
| 内存 | 随分辨率增长 | 固定（网络参数） |
| 查询速度 | 快（直接索引） | 慢（前向推理） |
| 连续性 | 离散 | 连续 |
| 可微分 | 离散梯度 | 天然可微 |

## 2. 坐标作为输入

### 2.1 基本形式

最简单的 INR：直接将坐标输入 MLP

```python
import torch
import torch.nn as nn

class SimpleINR(nn.Module):
    """最简单的隐式神经表示"""
    def __init__(self, in_dim=2, out_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid()
        )
    
    def forward(self, coords):
        # coords: (B, 2) 归一化坐标 [-1, 1]
        return self.net(coords)
```

### 2.2 问题：频谱偏差 (Spectral Bias)

直接用MLP学习高频信号非常困难。实验发现，标准MLP倾向于先学习低频分量，难以捕捉细节。

这被称为**频谱偏差**或**频率偏差**：网络对低频函数的拟合优先于高频。

## 3. 位置编码 (Positional Encoding)

### 3.1 傅里叶特征映射

NeRF 使用的**位置编码**将低维坐标映射到高维空间：

$$\gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \sin(2^1\pi p), \cos(2^1\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right)$$

- 对于3D位置 $\mathbf{p} = (x, y, z)$，$L=10$：输出维度 = $3 \times 2 \times 10 = 60$
- 对于3D方向 $\mathbf{d} = (\theta, \phi)$，$L=4$：输出维度 = $3 \times 2 \times 4 = 24$

```python
class PositionalEncoding(nn.Module):
    """傅里叶位置编码"""
    def __init__(self, in_dim, num_freqs, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.in_dim = in_dim
        
        # 频率: 2^0, 2^1, ..., 2^(L-1)
        self.freqs = 2.0 ** torch.arange(num_freqs).float()
        
        # 输出维度
        self.out_dim = in_dim * (1 + 2 * num_freqs) if include_input else in_dim * 2 * num_freqs
    
    def forward(self, x):
        # x: (..., in_dim)
        out = [x] if self.include_input else []
        for freq in self.freqs:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

# 使用示例
pe = PositionalEncoding(in_dim=3, num_freqs=10)
coords = torch.randn(1000, 3)
encoded = pe(coords)  # (1000, 63) = 3 + 3*2*10
```

### 3.2 为什么位置编码有效

从频率角度理解：
- 低频分量（$2^0, 2^1, \ldots$）捕获全局结构
- 高频分量（$2^{L-1}$）捕获精细细节
- 位置编码将高频信息显式注入网络，绕过了频谱偏差

**万能逼近定理的扩展**：带位置编码的MLP可以以任意精度逼近连续函数。

## 4. SIREN (Sinusoidal Representation Networks)

### 4.1 核心思想

SIREN (Sitzmann et al., 2020) 用**正弦激活函数**替代ReLU，从根本上解决频谱偏差：

$$\mathbf{x}_{l+1} = \sin(\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l)$$

### 4.2 初始化策略

SIREN 需要特殊的初始化保证信号的有效传播：

$$\mathbf{W}_l \sim \mathcal{U}\left(-\frac{\sqrt{6}}{n_l}, \frac{\sqrt{6}}{n_l}\right)$$

第一层需要更大的权重来捕获高频：

$$\mathbf{W}_0 \sim \mathcal{U}\left(-\frac{1}{n_0}, \frac{1}{n_0}\right) \times \omega_0$$

其中 $\omega_0 = 30$ 是频率缩放因子。

```python
class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, hidden=256, num_layers=5, omega_0=30):
        super().__init__()
        layers = [SirenLayer(in_dim, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden, hidden, omega_0=omega_0))
        layers.append(SirenLayer(hidden, out_dim, omega_0=1.0))
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords):
        return self.net(coords)
```

### 4.3 SIREN 的优势

1. **天然适合高频信号**：正弦激活可以表示任意频率
2. **隐式微分**：输出对输入的导数也有意义（如SDF的梯度 = 法线）
3. **可表示复杂信号**：图像、音频、3D形状、物理场

```python
# SIREN 的梯度 = 表面法线
def compute_sdf_normals(sdf_network, points):
    """计算SDF的梯度（即表面法线）"""
    points.requires_grad_(True)
    sdf = sdf_network(points)
    normals = torch.autograd.grad(
        outputs=sdf,
        inputs=points,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True
    )[0]
    return normals  # SDF梯度方向就是法线方向
```

## 5. 其他编码方式

### 5.1 球谐编码 (Spherical Harmonics)

用于表示方向相关的颜色（NeRF中使用）：

$$Y_l^m(\theta, \phi) = N_l^m P_l^m(\cos\theta) e^{im\phi}$$

### 5.2 学习型编码

用可学习的嵌入矩阵替代固定的傅里叶编码：

```python
class LearnedEncoding(nn.Module):
    def __init__(self, in_dim, num_bases, hidden_dim):
        super().__init__()
        self.basis = nn.Parameter(torch.randn(num_bases, in_dim))
        self.coeff = nn.Parameter(torch.randn(num_bases))
    
    def forward(self, x):
        # x: (..., in_dim)
        # 计算与每个基的内积
        proj = x @ self.basis.T  # (..., num_bases)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) * self.coeff
```

## 6. 应用场景

### 6.1 3D形状表示

用SDF网络表示3D形状：

$$f_\theta(x, y, z) = s \quad \text{where} \quad s = \begin{cases} < 0 & \text{内部} \\ = 0 & \text{表面} \\ > 0 & \text{外部} \end{cases}$$

### 6.2 图像表示

INR 可以用极少量参数表示高分辨率图像：

```python
# 用SIREN表示图像
model = SIREN(in_dim=2, out_dim=3, hidden=256, num_layers=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 坐标网格
H, W = 512, 512
y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
coords = torch.stack([x, y], dim=-1).reshape(-1, 2)

# 训练
for step in range(5000):
    pred = model(coords)
    loss = F.mse_loss(pred, image_flat)  # image_flat: 原图展平
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 6.3 物理仿真

INR 可表示物理场（压力场、速度场、温度场），实现连续分辨率的仿真。

## 7. INR vs 传统表示

| 场景 | 传统方法 | INR |
|------|----------|-----|
| 3D形状 | 体素/网格 | DeepSDF, Occupancy Networks |
| 新视角合成 | MVS + 纹理 | NeRF |
| 图像 | 像素网格 | SIREN图像 |
| 物理场 | 有限元网格 | 神经PDE |

---

**关键要点**：
1. INR用坐标查询网络获取信号值，实现了连续、紧凑的信号表示
2. 频谱偏差是标准MLP的固有问题，位置编码和SIREN是两种解决方案
3. SIREN用正弦激活和特殊初始化，天然适合高频信号和隐式微分
4. INR 是 NeRF、DeepSDF 等后续工作的基础
