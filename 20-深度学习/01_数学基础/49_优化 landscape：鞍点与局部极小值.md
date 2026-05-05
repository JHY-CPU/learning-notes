# 50_优化 landscape：鞍点与局部极小值

## 核心概念

- **优化景观 (Optimization Landscape)**：损失函数 $L(\theta)$ 在参数空间 $\Theta$ 上的"地形图"，包含山峰（局部极大值）、山谷（局部极小值）和鞍部（鞍点）。景观的几何特性决定了优化算法的难度和路径。
- **局部极小值 (Local Minimum)**：$\theta^*$ 的邻域内所有点都有 $L(\theta) \geq L(\theta^*)$。必要条件：$\nabla L = 0$，$\nabla^2 L \succeq 0$（Hessian 半正定）。
- **局部极大值 (Local Maximum)**：$\nabla L = 0$，$\nabla^2 L \preceq 0$（Hessian 半负定）。
- **鞍点 (Saddle Point)**：$\nabla L = 0$，但 Hessian 既有正特征值也有负特征值。沿着正特征值方向函数上升，负特征值方向下降。鞍点是高维非凸优化中的主要障碍。
- **平坦区域 (Flat Region)**：梯度接近零但 Hessian 特征值也很小的区域。优化在此几乎停滞（梯度消失），但尚未达到真正的临界点。
- **深渊 (Ravine)**：一个方向曲率大（陡峭）、另一个方向曲率小（平缓）的区域。梯度下降在此会沿平缓方向缓慢下降而在陡峭方向来回振荡。
- **经验观察**：在高维深度学习中，大多数局部极小值在损失值上接近全局极小值，且鞍点而非局部极小值是更严重的障碍。

## 数学推导

临界点的一阶条件：
$$
\nabla L(\theta^*) = 0
$$

局部极小值的二阶充分条件：
$$
\nabla L(\theta^*) = 0, \quad \nabla^2 L(\theta^*) \succ 0 \quad (\text{正定})
$$

鞍点的判定：
$$
\nabla L(\theta^*) = 0, \quad \lambda_{\min}(\nabla^2 L(\theta^*)) < 0 < \lambda_{\max}(\nabla^2 L(\theta^*))
$$

标准梯度下降在鞍点附近的行为：
$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) \approx \theta_t - \eta H (\theta_t - \theta^*)
$$

在特征向量基下，各分量独立演化：
$$
\theta_t^{(i)} - \theta^{*(i)} = (1 - \eta \lambda_i)^t (\theta_0^{(i)} - \theta^{*(i)})
$$

当 $\lambda_i < 0$ 时，$1 - \eta\lambda_i > 1$，分量沿该方向发散，从鞍点逃离。
当 $\lambda_i > 0$ 时，$|1 - \eta\lambda_i| < 1$，分量收敛到鞍点。

这说明鞍点就像"马鞍"——沿着负曲率方向可以逃离，沿正曲率方向被吸引。

## 直观理解

- **真实山地的类比**：优化 landscape 就像真实的山地。局部极小值是小山谷，鞍点是山垭口（一边上坡、一边下坡）。在高维空间中，大部分"平坦点"是鞍点而非山谷——这就像在高山上，大部分平坦区域是山脊而非盆地底部。
- **为什么高维中鞍点远多于局部极小值**：直观来说，随机 Hessian 矩阵的特征值有正有负的概率远大于全正或全负。在 $d$ 维中，局部极小值在所有方向都是"上坡"——概率约为 $2^{-d}$，随维度指数衰减。而鞍点几乎无处不在。
- **牛顿法在鞍点的问题**：牛顿法 $\theta_{t+1} = \theta_t - H^{-1} \nabla L$ 在鞍点附近会被负曲率方向"欺骗"——它会把梯度指向负特征值方向，但实际应该沿该方向下降（如果目标是找极小值）。因此牛顿法在鞍点处会被吸引而非逃离。

## 代码示例

```python
import numpy as np

# 1. 鞍点 vs 局部极小值
def saddle_function(x, y):
    """典型鞍点函数 f(x,y) = x^2 - y^2"""
    return x**2 - y**2

def local_min_function(x, y):
    """局部极小值 f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def saddle_grad(x, y):
    return np.array([2*x, -2*y])

def local_min_grad(x, y):
    return np.array([2*x, 2*y])

def saddle_hessian(x, y):
    return np.array([[2, 0], [0, -2]])

def local_min_hessian(x, y):
    return np.array([[2, 0], [0, 2]])

print("临界点分析:")
print(f"  鞍点 Hessian 特征值: {np.linalg.eigvalsh(saddle_hessian(0,0))}")
print(f"  局部极小 Hessian 特征值: {np.linalg.eigvalsh(local_min_hessian(0,0))}")

# 2. 梯度下降在鞍点附近的行为
def gradient_descent_landscape(grad_func, x0, y0, lr=0.1, n_steps=20):
    x, y = x0, y0
    path = [(x, y)]
    values = [saddle_function(x, y)]
    for _ in range(n_steps):
        g = grad_func(x, y)
        x = x - lr * g[0]
        y = y - lr * g[1]
        path.append((x, y))
        values.append(saddle_function(x, y))
    return np.array(path), np.array(values)

# 从不同起点开始
print("\n梯度下降在鞍点附近的行为:")
for start in [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]:
    path, values = gradient_descent_landscape(saddle_grad, start[0], start[1], lr=0.1)
    end = path[-1]
    print(f"  起点 {start}: 终点 ({end[0]:.4f}, {end[1]:.4f}), "
          f"最终值={values[-1]:.4f}, 是否收敛到鞍点? {np.allclose(end, [0, 0], atol=1)}")

# 3. 随机梯度噪声帮助逃离鞍点
np.random.seed(42)

def sgd_with_noise(grad_func, x0, y0, lr=0.1, noise_std=0.05, n_steps=50):
    x, y = x0, y0
    path = [(x, y)]
    for _ in range(n_steps):
        g = grad_func(x, y)
        # 添加噪声
        noise = noise_std * np.random.randn(2)
        x = x - lr * g[0] + noise[0]
        y = y - lr * g[1] + noise[1]
        path.append((x, y))
    return np.array(path)

print("\nSGD 噪声帮助逃离鞍点:")
path_sgd = sgd_with_noise(saddle_grad, 0.1, 0.1, lr=0.1, noise_std=0.08, n_steps=100)
end = path_sgd[-1]
print(f"  起点 (0.1, 0.1), 终点 ({end[0]:.4f}, {end[1]:.4f})")
print(f"  鞍点在 (0,0), 逃离? {np.linalg.norm(end) > 0.5}")

# 4. Hessian 特征值分布分析
# 模拟高维随机函数的 Hessian
def random_hessian_spectrum(dim=100, n_instances=1000):
    neg_eigvals = 0
    pos_eigvals = 0
    mixed = 0
    for _ in range(n_instances):
        H = np.random.randn(dim, dim)
        H = H.T @ H  # 随机半正定
        H = H - np.mean(np.linalg.eigvalsh(H)) * np.eye(dim)  # 平移使有正有负
        eigvals = np.linalg.eigvalsh(H)
        has_pos = np.any(eigvals > 1e-6)
        has_neg = np.any(eigvals < -1e-6)
        if has_pos and has_neg:
            mixed += 1
        elif has_pos:
            pos_eigvals += 1
        else:
            neg_eigvals += 1
    return pos_eigvals, neg_eigvals, mixed

print(f"\n高维 Hessian 特征值分布 (dim=50):")
pos, neg, mixed = random_hessian_spectrum(50, 500)
print(f"  正定: {pos} ({pos/5:.1f}%), 负定: {neg} ({neg/5:.1f}%), 鞍点: {mixed} ({mixed/5:.1f}%)")

# 5. 损失景观的"谷底"宽度分析
def loss_landscape_profile(H, direction, center=0):
    """沿某方向的损失剖面"""
    xs = np.linspace(-5, 5, 100)
    losses = [center + 0.5 * x**2 * (direction.T @ H @ direction) for x in xs]
    return xs, losses

# 窄谷 vs 宽谷
H_narrow = np.array([[100, 0], [0, 1]])  # 条件数 100
H_wide = np.array([[1, 0], [0, 1]])  # 条件数 1
print(f"\n窄谷 (条件数=100) vs 宽谷 (条件数=1):")
print(f"  窄谷: 沿 x 方向曲率大 (λ=100), 沿 y 方向曲率小 (λ=1)")
print(f"  宽谷: 各向同性 (λ=1,1)")

# 6. 不同算法在窄谷上的表现
def gradient_descent(grad, x0, lr, n_iter):
    x = x0.copy()
    for _ in range(n_iter):
        x = x - lr * grad(x)
    return x

def momentum_gd(grad, x0, lr, mu=0.9, n_iter=50):
    x = x0.copy()
    v = np.zeros_like(x)
    for _ in range(n_iter):
        v = mu * v + lr * grad(x)
        x = x - v
    return x

# 二次型 f(x,y) = 100x^2 + y^2
H = np.array([[200, 0], [0, 2]])  # Hessian 的 2 倍
grad_quad = lambda x: H @ x

x0 = np.array([1.0, 1.0])
gd_end = gradient_descent(grad_quad, x0, lr=0.005, n_iter=100)
mom_end = momentum_gd(grad_quad, x0, lr=0.005, mu=0.9, n_iter=100)

print(f"\n窄谷优化 (100x² + y²):")
print(f"  GD 终点: ({gd_end[0]:.4f}, {gd_end[1]:.4f})")
print(f"  Momentum 终点: ({mom_end[0]:.4f}, {mom_end[1]:.4f})")
```

## 深度学习关联

- **为什么深度网络能找到好的局部极小值**：传统观点认为非凸优化很困难，但深度学习的实践反复证明 SGD 能找到泛化好的解。理论研究表明，在过度参数化（overparameterized）的网络中，所有局部极小值都是全局极小值。这是因为过度参数化使损失函数的 Hessian 在最小的点处几乎处处正定，减少了鞍点和不良局部极小值的数量。
- **Flat Minima vs Sharp Minima**：实验观察到，SGD 倾向于收敛到"平坦的极小值"（Hessian 特征值小），而"尖锐的极小值"（Hessian 特征值大）泛化性能差。这是因为平坦极小值对参数扰动的敏感性较低——测试分布的小偏移不会导致性能大幅下降。这一发现促使了 Sharpness-Aware Minimization (SAM) 等显式寻找平坦极小值的优化方法。
- **SGD 的隐式正则化**：为什么 SGD 能找到好的解？一个重要原因是 SGD 的梯度噪声本身就是一个正则化器。噪声的大小自适应于 landscape 的曲率——在尖锐方向噪声相对较大（帮助逃离），在平坦方向噪声相对较小（帮助收敛）。SGD 的隐式正则化效应可以解释为一种"扩散过程"，其稳态分布偏向于"宽谷"的底部。
- **批大小与景观探索**：小批大小引入更多梯度噪声，有助于逃离鞍点和尖锐极小值，但可能无法精细收敛。大批大小梯度更准确，但容易被鞍点困住或收敛到尖锐极小值。"批大小 warmup"策略（从小批大小开始，逐渐增大）结合了两者的优势——早期进行广泛探索，后期精细收敛。
