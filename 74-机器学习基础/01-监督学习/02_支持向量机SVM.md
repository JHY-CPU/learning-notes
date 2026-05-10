# 支持向量机SVM - 机器学习基础


# 支持向量机 SVM


机器学习基础 - 监督学习 | 最大间隔分类器与核方法


## 目录


1. [SVM 概述与核心思想](#intro)
2. [线性 SVM 与最大间隔](#linear-svm)
3. [软间隔 SVM](#soft-margin)
4. [核函数与核技巧](#kernel)
5. [SMO 算法](#smo)
6. [代码实现](#code)
7. [总结](#summary)


## 1. SVM 概述与核心思想


支持向量机（Support Vector Machine, SVM）是一种强大的监督学习算法，可用于分类和回归任务。其核心思想是找到一个**最优超平面**，使得不同类别之间的**间隔（margin）最大化**。


### 1.1 核心概念


- **超平面（Hyperplane）**
   ：在 n 维空间中，超平面是 n-1 维的子空间。二维空间中是直线，三维空间中是平面
- **间隔（Margin）**
   ：超平面到最近样本点（支持向量）的距离的两倍
- **支持向量（Support Vectors）**
   ：距离超平面最近的那些样本点，它们决定了超平面的位置


### 1.2 SVM 的优势


- 在高维空间中表现优异
- 当数据维度大于样本数时仍然有效
- 只使用支持向量做决策，内存效率高
- 核技巧使其能处理非线性问题


## 2. 线性 SVM 与最大间隔


### 2.1 数学表述


对于线性可分的数据，SVM 寻找超平面 wᵀx + b = 0，使得：


$$
y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1, ∀i
$$


即所有正类样本满足 wᵀx + b ≥ 1，负类样本满足 wᵀx + b ≤ -1。


### 2.2 最大间隔优化问题


间隔宽度为 2/||w||，最大化间隔等价于最小化 ||w||²/2：


$$
min (1/2)||w||²  s.t.  y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1, ∀i
$$


### 2.3 拉格朗日对偶问题


引入拉格朗日乘子 αᵢ ≥ 0，原始问题的对偶形式为：


$$
max Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼy⁽ⁱ⁾y⁽ʲ⁾x⁽ⁱ⁾ᵀx⁽ʲ⁾
                s.t. αᵢ ≥ 0, Σᵢ αᵢy⁽ⁱ⁾ = 0
$$


对偶问题的关键优势：


- 数据以
   **内积形式**
   出现，便于使用核技巧
- 大部分 αᵢ = 0，只有支持向量对应的 αᵢ > 0
- 问题规模只与支持向量数量有关


### 2.4 KKT 条件


最优解必须满足 KKT 条件：


$$
αᵢ ≥ 0
                y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) - 1 ≥ 0
                αᵢ[y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) - 1] = 0
$$


最后一个互补松弛条件说明：只有支持向量（在间隔边界上的点）才满足 αᵢ > 0，其余点 αᵢ = 0。


## 3. 软间隔 SVM


现实数据往往不是线性可分的。软间隔 SVM 允许部分样本违反间隔约束，但对违规样本施加惩罚。


### 3.1 引入松弛变量


$$
min (1/2)||w||² + C Σᵢ ξᵢ
                s.t. y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0, ∀i
$$


其中：


- **ξᵢ**
   ：松弛变量，表示第 i 个样本违反间隔约束的程度
- **C**
   ：惩罚参数，控制"容忍违规"与"保持大间隔"之间的权衡
- C 越大 → 对违规惩罚越重 → 间隔越窄 → 可能过拟合
- C 越小 → 更容忍违规 → 间隔越宽 → 可能欠拟合


### 3.2 合页损失函数


软间隔 SVM 等价于最小化合页损失（Hinge Loss）：


$$
L(y, f(x)) = max(0, 1 - y·f(x))
$$


当 y·f(x) ≥ 1 时损失为 0（分类正确且在间隔外），否则损失线性增长。


### 3.3 对偶问题的变化


软间隔的对偶问题与硬间隔几乎相同，只是 αᵢ 的约束变为有界：


$$
0 ≤ αᵢ ≤ C, ∀i
$$


### 3.4 C 的选择


| C 值 | 效果 | 适用场景 |
| --- | --- | --- |
| 很小 (如 0.01) | 间隔宽，允许更多误分类 | 噪声多的数据 |
| 适中 (如 1.0) | 平衡间隔与误分类 | 一般场景 |
| 很大 (如 100) | 间隔窄，尽量正确分类 | 数据质量高 |


## 4. 核函数与核技巧


核技巧是 SVM 最强大的特性之一，它使得 SVM 能在高维特征空间中隐式地计算，而无需显式地将数据映射到高维空间。


### 4.1 核技巧的原理


对偶问题中数据以内积 x⁽ⁱ⁾ᵀx⁽ʲ⁾ 形式出现。核函数 K(x⁽ⁱ⁾, x⁽ʲ⁾) = φ(x⁽ⁱ⁾)ᵀφ(x⁽ʲ⁾) 直接在原始空间计算高维空间的内积，避免了显式映射 φ 的计算。


### 4.2 常用核函数


| 核函数 | 公式 | 特点 | 适用场景 |
| --- | --- | --- | --- |
| **线性核** | K(x,y) = xᵀy | 最简单，等价于线性 SVM | 线性可分数据、高维稀疏数据 |
| **多项式核** | K(x,y) = (γxᵀy + r)ᵈ | d 控制多项式次数 | NLP、图像处理 |
| **RBF 核（高斯核）** | K(x,y) = exp(-γ\|\|x-y\|\|²) | 最常用，可映射到无穷维 | 通用场景，默认首选 |
| **Sigmoid 核** | K(x,y) = tanh(γxᵀy + r) | 类似神经网络 | 较少使用 |


### 4.3 RBF 核的参数 γ


RBF 核 K(x,y) = exp(-γ||x-y||²) 中的 γ 参数控制高斯函数的"宽度"：


- **γ 大**
   ：高斯函数窄，只关注近邻点 → 决策边界复杂 → 容易过拟合
- **γ 小**
   ：高斯函数宽，关注更远的点 → 决策边界平滑 → 可能欠拟合


> **Tip:** **核函数选择建议：**
>
> - 优先尝试 RBF 核（默认选择）
> - 特征数 >> 样本数，用线性核
> - 使用交叉验证选择 C 和 γ
> - 可以用网格搜索 GridSearchCV 寻找最优参数组合


### 4.4 核函数的条件（Mercer 定理）


一个有效的核函数必须满足：对于任意数据集，核矩阵 K（K_ij = K(x⁽ⁱ⁾, x⁽ʲ⁾)）是半正定矩阵。


## 5. SMO 算法


序列最小优化（Sequential Minimal Optimization, SMO）是求解 SVM 对偶问题的高效算法，由 John Platt 于 1998 年提出。


### 5.1 核心思想


SMO 将大的优化问题分解为一系列最小的子问题：每次只优化**两个**拉格朗日乘子 αᵢ 和 αⱼ，保持其他乘子不变。


选择两个而非一个的原因：约束 Σᵢ αᵢy⁽ⁱ⁾ = 0 意味着至少需要同时更新两个乘子才能保持等式约束。


### 5.2 算法步骤


1. 选择一对乘子 (αᵢ, αⱼ)
2. 固定其他乘子，将目标函数表示为这两个乘子的函数
3. 解析求解这两个乘子的最优值
4. 更新阈值 b
5. 重复直到收敛（所有乘子满足 KKT 条件）


### 5.3 乘子更新公式


设 Eᵢ = f(x⁽ⁱ⁾) - y⁽ⁱ⁾ 为预测误差，更新后的 αⱼ 为：


$$
αⱼ_new = αⱼ_old + y⁽ʲ⁾(Eᵢ - Eⱼ) / η
                其中 η = 2K(x⁽ⁱ⁾, x⁽ʲ⁾) - K(x⁽ⁱ⁾, x⁽ⁱ⁾) - K(x⁽ʲ⁾, x⁽ʲ⁾)
$$


裁剪到 [L, H] 范围后，再根据约束更新 αᵢ。


### 5.4 启发式选择策略


- **第一个乘子**
   ：选择违反 KKT 条件最严重的乘子
- **第二个乘子**
   ：选择使 |Eᵢ - Eⱼ| 最大的乘子（期望最大的参数更新）


## 6. Python 代码实现


### 6.1 使用 scikit-learn


```
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# 数据预处理 - SVM 对特征尺度敏感
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 线性 SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF 核 SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# 网格搜索找最优参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"最优参数: {grid.best_params_}")
print(f"最优得分: {grid.best_score_:.4f}")

# 查看支持向量
print(f"支持向量数量: {len(svm_rbf.support_vectors_)}")
print(f"各类支持向量数: {svm_rbf.n_support_}")
```


### 6.2 简化版 SMO 实现


```
import numpy as np

class SimpleSVM:
    def __init__(self, C=1.0, tol=0.001, max_iter=100):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def _kernel(self, X):
        # 线性核矩阵
return np.dot(X, X.T)

    def fit(self, X, y):
        n = X.shape[0]
        self.alpha = np.zeros(n)
        self.b = 0
        K = self._kernel(X)

        for iteration in range(self.max_iter):
            changed = 0
for i in range(n):
                Ei = np.sum(self.alpha * y * K[i]) + self.b - y[i]
                if (y[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i]*Ei > self.tol and self.alpha[i] > 0):
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    Ej = np.sum(self.alpha * y * K[j]) + self.b - y[j]
                    # 简化的更新逻辑
                    eta = 2*K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0: continue
                    L = max(0, self.alpha[j] - self.alpha[i]) if y[i] != y[j] \
                        else max(0, self.alpha[i]+self.alpha[j]-self.C)
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i]) if y[i] != y[j] \
                        else min(self.C, self.alpha[i]+self.alpha[j])
                    if L == H: continue
                    self.alpha[j] -= y[j]*(Ei-Ej)/eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    changed += 1
if changed == 0: break
def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
```


## 7. 总结


### SVM 核心要点


1. **最大间隔**
   是 SVM 的核心思想，通过寻找最优超平面实现泛化能力
2. **支持向量**
   是决定分类边界的关键样本，大部分训练数据对模型无影响
3. **软间隔**
   通过参数 C 控制过拟合与欠拟合的平衡
4. **核技巧**
   使 SVM 能处理非线性问题，无需显式特征映射
5. **SMO 算法**
   高效求解 SVM 优化问题


### SVM vs 其他算法


| 对比维度 | SVM | 逻辑回归 | 决策树 |
| --- | --- | --- | --- |
| 决策边界 | 最大间隔超平面 | 概率分界面 | 轴平行分割 |
| 非线性处理 | 核技巧 | 特征工程 | 天然支持 |
| 大数据集 | 较慢（O(n²~n³)） | 较快 | 较快 |
| 可解释性 | 较低（核方法） | 高 | 高 |
| 过拟合风险 | 低（大间隔） | 中等 | 较高 |


> **Tip:** **实践建议：**
> SVM 在中小规模数据集（<10万样本）和高维特征空间中表现优异。对于大规模数据，考虑使用线性 SVM 或转向树模型/神经网络。训练前务必进行特征标准化。


机器学习基础笔记 - 监督学习 - 支持向量机SVM


内容涵盖：最大间隔、软间隔、核函数(RBF/多项式)、SMO算法、网格搜索调参


<!-- Converted from: 02_支持向量机SVM.html -->
