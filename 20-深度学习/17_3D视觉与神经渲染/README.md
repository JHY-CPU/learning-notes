# 3D视觉与神经渲染

## 一、3D表示

### 1.1 表示方式

| 表示 | 说明 | 优缺点 |
|------|------|--------|
| 点云 | 无序3D点集合 | 简单但不连续 |
| 网格 | 顶点+面片 | 可渲染但拓扑固定 |
| 体素 | 3D网格 | 规则但分辨率受限 |
| 隐式场 | 连续函数 $f(x,y,z)=sdf$ | 任意分辨率但慢 |
| NeRF | 辐射场 | 高质量但训练慢 |

---

## 二、NeRF

### 2.1 核心思想

用神经网络表示3D场景的辐射场：
$$F_\Theta: (x, y, z, \theta, \phi) \rightarrow (r, g, b, \sigma)$$

输入位置和方向，输出颜色和体密度。

### 2.2 体渲染

沿光线积分：
$$C(r) = \int_{t_n}^{t_f} T(t) \cdot \sigma(t) \cdot c(t) \, dt$$

其中 $T(t) = \exp(-\int_{t_n}^{t} \sigma(s) ds)$ 为透射率。

### 2.3 改进

- **Instant-NGP**：哈希编码加速训练
- **3D Gaussian Splatting**：显式高斯表示，实时渲染
- **Mip-NeRF**：解决抗锯齿问题
- **TensoRF**：张量分解加速

---

## 三、3D重建

### 3.1 传统方法

- **SfM**（Structure from Motion）：从多视图恢复3D结构
- **MVS**（Multi-View Stereo）：稠密重建

### 3.2 深度学习方法

- **单目深度估计**：从单张图估计深度
- **多视图重建**：MVSNet
- **神经隐式重建**：NeuS、VolSDF

---

## 四、3D生成

- **Point-E / Shap-E**：OpenAI的3D生成
- **DreamFusion**：用SDS损失从2D扩散生成3D
- **Zero-1-to-3**：从单图生成新视角
- **Wonder3D**：从单图生成3D纹理网格
