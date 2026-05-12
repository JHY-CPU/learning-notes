# SVD与Jordan标准形


## 一、LU 分解（预备知识）


将矩阵 A 分解为下三角矩阵 L 与上三角矩阵 U 的乘积：


$$
A = LU
$$


其中 L 是单位下三角矩阵（对角线全为1），U 是上三角矩阵。

**带行交换的 LU 分解：**
PA = LU


P 是置换矩阵，L 是单位下三角矩阵，U 是上三角矩阵。


对应高斯消元法中行交换 + 消元的过程。

## 二、QR 分解


将矩阵 A 分解为正交矩阵 Q 与上三角矩阵 R 的乘积：


$$
A = QR
$$


其中 Q 的各列由 A 的列向量经 Gram-Schmidt 正交化得到。

**应用：**


- 求解线性方程组（避免直接求逆）


- 计算特征值（QR 迭代算法）


- 最小二乘问题

## 三、奇异值分解（SVD） 核心重点


任意 m×n 实矩阵 A 都可以分解为：


$$
A = UΣVᵀ
$$


| 矩阵 | 维度 | 性质 | 含义 |
| --- | --- | --- | --- |
| U | m×m | 正交矩阵（UᵀU = I） | AAᵀ 的特征向量组成的矩阵（左奇异向量） |
| Σ | m×n | 对角矩阵，元素 ≥ 0 | 奇异值 σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 |
| V | n×n | 正交矩阵（VᵀV = I） | AᵀA 的特征向量组成的矩阵（右奇异向量） |


### 奇异值的计算


$$
σᵢ = √(λᵢ)
        其中 λᵢ 是 AᵀA（或 AAᵀ）的非零特征值
$$


### SVD 的几何意义

任何线性变换 = 旋转(Vᵀ) → 伸缩(Σ) → 旋转(U)


即：先旋转到标准坐标系，再沿各轴伸缩，最后再旋转。

### SVD 的应用

**1. 伪逆（Moore-Penrose）：**
A⁺ = VΣ⁺Uᵀ（Σ⁺ 将非零奇异值取倒数并转置）


**2. 数据压缩：**
保留前 k 个最大奇异值，A ≈ UₖΣₖVₖᵀ（截断SVD）


**3. 推荐系统：**
矩阵补全


**4. 降维（PCA）：**
数据的主成分分析

## 四、Jordan 标准形 理论重点


若 A 不能对角化，则存在可逆矩阵 P 使得：


$$
P⁻¹AP = J = diag(J₁, J₂, ..., Jₖ)
$$


其中 J 是 Jordan 标准形（分块对角矩阵），每个 Jᵢ 是一个 Jordan 块。


### Jordan 块


特征值为 λ 的 r 阶 Jordan 块：


$$
Jᵢ(λ) =

λ
1
0
0
0
λ
1
0
0
0
λ
1
0
0
0
λ
$$

**Jordan 块的特点：**


- 对角线全是特征值 λ


- 主对角线上方一个位置全是 1（次对角线）


- 其余位置全是 0


- 特征值 λ 的 Jordan 块的个数 = dim(Eλ)（几何重数）

### 重要性质

- Jordan 标准形在不计排列顺序下是
**唯一**
的


- A 可对角化 ⟺ A 的所有 Jordan 块都是一阶的


- A 的最小多项式 = 各特征值最大 Jordan 块阶数对应的因式之积


- 特征值 λ 的最大 Jordan 块阶数 = λ 的指数（使 (A-λI)ᵏ = 0 的最小 k）

> **Warning:** **考试提示：**
> Jordan 标准形的计算较复杂，重点掌握概念和性质。理解"不可对角化时的最简形式"这一核心思想。


## 五、矩阵分解方法对比


| 分解 | 形式 | 适用范围 | 主要用途 |
| --- | --- | --- | --- |
| LU | A = LU | 方阵（可逆或不可逆） | 解线性方程组 |
| QR | A = QR | 任意矩阵 | 特征值计算、最小二乘 |
| 对角化 | A = PΛP⁻¹ | 可对角化的方阵 | 矩阵幂、矩阵函数 |
| SVD | A = UΣVᵀ | 任意矩阵 | 降维、伪逆、数据压缩 |
| Jordan | A = PJP⁻¹ | 方阵 | 理论分析、不可对角化时 |
| 正交对角化 | A = QΛQᵀ | 实对称矩阵 | 二次型、正交变换 |


## Python实现

### SVD 奇异值分解

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

# SVD 分解: A = U Σ V^T
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U 的形状: {U.shape}")       # (4, 3)
print(f"奇异值 σ: {sigma}")          # 降序排列
print(f"V^T 的形状: {Vt.shape}")    # (3, 3)

# 验证 A = U Σ V^T
Sigma = np.diag(sigma)
A_reconstructed = U @ Sigma @ Vt
print(f"A 重构验证: {np.allclose(A, A_reconstructed)}")

# 用 SVD 计算矩阵的秩
rank = np.sum(sigma > 1e-10)
print(f"矩阵的秩 (通过SVD): {rank}")
```

### SVD 数据压缩

```python
import numpy as np

# 用前 k 个奇异值近似矩阵
def svd_compress(A, k):
    """用前 k 个奇异值做截断 SVD 压缩"""
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    A_approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    compression_ratio = (U.shape[0] * k + k + k * Vt.shape[1]) / (A.shape[0] * A.shape[1])
    error = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
    return A_approx, error, compression_ratio

A = np.random.randn(100, 50)
for k in [1, 5, 10, 20]:
    _, error, ratio = svd_compress(A, k)
    print(f"k={k:2d}: 相对误差={error:.4f}, 压缩率={ratio:.2%}")
```

### LU 分解与 QR 分解

```python
import numpy as np
from scipy.linalg import lu, qr

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# LU 分解: PA = LU
P, L, U = lu(A)
print(f"L =\n{np.round(L, 4)}")
print(f"U =\n{np.round(U, 4)}")
print(f"PA = LU 验证: {np.allclose(P @ A, L @ U)}")

# QR 分解: A = QR
Q, R = qr(A)
print(f"\nQ =\n{np.round(Q, 4)}")
print(f"R =\n{np.round(R, 4)}")
print(f"Q 正交验证: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"A = QR 验证: {np.allclose(A, Q @ R)}")
```

<!-- Converted from: 03_SVD与Jordan标准形.html -->
