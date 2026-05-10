# 动量法与Nesterov加速 — 最优化理论笔记


## 一、SGD的问题：为什么需要动量？


标准SGD存在两个主要问题：


### 1.1 震荡问题


- 在狭长的"峡谷"形损失面上，梯度在窄方向上分量大，导致来回震荡
- 为避免发散，必须使用很小的学习率，导致沿峡谷方向（真正需要前进的方向）收敛很慢
- 数学上：Hessian矩阵的条件数 \( \kappa = L/\mu \) 很大时，SGD收敛速度退化为 \( O((\kappa-1)/(\kappa+1))^k \)


### 1.2 收敛速度慢


- 即使在简单凸问题上，SGD的收敛率也只有 \( O(1/k) \)
- 高曲率方向需要很小的学习率，限制了所有方向上的步长


> **Note:** **物理直觉：**
> 想象一个小球在山谷中滚动。纯SGD就像小球没有惯性，每一步完全由当前坡度决定，会在山谷两侧来回弹跳。而动量法给小球加上了惯性，让它沿着一致的方向加速。


## 二、动量法（Momentum）


### 2.1 核心思想


引入动量变量 \( m_t \)（速度），累积历史梯度信息：


$$
\(m_t = \beta m_{t-1} + \nabla f(x_t)\)
                \(x_{t+1} = x_t - \alpha m_t\)
$$


其中 \( \beta \in [0, 1) \) 是动量系数，通常取 \( \beta = 0.9 \)。


### 2.2 展开形式


将递推展开，动量项是历史梯度的**指数加权移动平均**：


$$
\(m_t = \sum_{\tau=0}^{t} \beta^\tau \nabla f(x_{t-\tau})\)
$$


当 \( \beta = 0.9 \) 时，大约最近10步的梯度对当前动量贡献显著。


### 2.3 动量的作用


- **加速收敛：**
   在一致梯度方向上累积速度，步长越来越大
- **抑制震荡：**
   相反方向的梯度相互抵消，只保留一致分量
- **等效平均：**
   相当于对梯度做了平滑处理，减少了随机噪声的影响


> **Example:** **直观理解：**
>
>
> 当梯度方向一致时（如沿斜坡向下），\( m_t \) 不断累积增大，步长变大 → 加速。
>
>
> 当梯度方向交替变化时（如在峡谷中震荡），正负梯度部分抵消 → 减少震荡。
>
>
>
>
> 类比：球在斜坡上滚动，动量让球越滚越快，不会因为小的凹凸就停住。


## 三、Nesterov加速梯度（NAG）


### 3.1 动量法的不足


标准动量法在当前位置计算梯度，但动量已经让参数"往前走了一步"。当动量很大时，可能会"冲过头"。


### 3.2 Nesterov加速的思想


先按动量方向走一步，在"前瞻位置"计算梯度：


$$
\(m_t = \beta m_{t-1} + \nabla f(x_t - \alpha \beta m_{t-1})\)
                \(x_{t+1} = x_t - \alpha m_t\)
$$


等价形式（更常见）：


$$
\(\tilde{x}_t = x_t - \alpha \beta m_{t-1} \quad \text{（前瞻位置）}\)
                \(m_t = \beta m_{t-1} + \nabla f(\tilde{x}_t)\)
                \(x_{t+1} = x_t - \alpha m_t\)
$$


### 3.3 为什么NAG更好


- 在"前瞻位置"算梯度，相当于多看了一步，能更早地修正方向
- 减少了"冲过头"的可能性
- 对于凸函数，NAG的收敛率为 \( O(1/k^2) \)，而标准动量只有 \( O(1/k) \)
- 这个加速率是梯度方法在凸光滑问题上的理论最优


### 3.4 三种方法更新公式对比


| 方法 | 梯度计算位置 | 更新公式 | 收敛率（凸） |
| --- | --- | --- | --- |
| SGD | \( x_t \) | \( x_{t+1} = x_t - \alpha \nabla f(x_t) \) | \( O(1/k) \) |
| Momentum | \( x_t \) | \( m_t = \beta m_{t-1} + \nabla f(x_t) \)
\( x_{t+1} = x_t - \alpha m_t \) | \( O(1/k) \) |
| NAG | \( x_t - \alpha\beta m_{t-1} \) | \( m_t = \beta m_{t-1} + \nabla f(\tilde{x}_t) \)
\( x_{t+1} = x_t - \alpha m_t \) | \( O(1/k^2) \) |


> **Important:** **NAG的精髓：**
> 不是在当前点算梯度，而是先"预见"动量方向上的下一步，在那里算梯度。这给了优化器一个"纠错"的机会。


## 四、Python代码：实现Momentum和NAG


```
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# ==========================================
# SGD（无动量）
# ==========================================
def sgd(grad, x0, lr=0.001, n_iter=3000):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(n_iter):
        x = x - lr * grad(x)
        path.append(x.copy())
    return x, np.array(path)

# ==========================================
# Momentum SGD
# ==========================================
def momentum(grad, x0, lr=0.001, beta=0.9, n_iter=3000):
    x = x0.copy()
    m = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_iter):
        g = grad(x)
        m = beta * m + g
        x = x - lr * m
        path.append(x.copy())
    return x, np.array(path)

# ==========================================
# Nesterov Accelerated Gradient (NAG)
# ==========================================
def nesterov(grad, x0, lr=0.001, beta=0.9, n_iter=3000):
    x = x0.copy()
    m = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_iter):
        # 先按动量方向走一步，在前瞻位置计算梯度
        x_lookahead = x - lr * beta * m
        g = grad(x_lookahead)
        m = beta * m + g
        x = x - lr * m
        path.append(x.copy())
    return x, np.array(path)

# 运行三种方法
x0 = np.array([-1.0, 1.0])
x_sgd, path_sgd = sgd(rosenbrock_grad, x0, lr=0.001, n_iter=3000)
x_mom, path_mom = momentum(rosenbrock_grad, x0, lr=0.001, beta=0.9, n_iter=3000)
x_nag, path_nag = nesterov(rosenbrock_grad, x0, lr=0.001, beta=0.9, n_iter=3000)

print("=== 三种方法对比（Rosenbrock函数）===")
print(f"SGD:      f={rosenbrock(x_sgd):.6f}, x=({x_sgd[0]:.4f}, {x_sgd[1]:.4f})")
print(f"Momentum: f={rosenbrock(x_mom):.6f}, x=({x_mom[0]:.4f}, {x_mom[1]:.4f})")
print(f"NAG:      f={rosenbrock(x_nag):.6f}, x=({x_nag[0]:.4f}, {x_nag[1]:.4f})")

# 绘制收敛曲线
loss_sgd = [rosenbrock(p) for p in path_sgd]
loss_mom = [rosenbrock(p) for p in path_mom]
loss_nag = [rosenbrock(p) for p in path_nag]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 收敛曲线
axes[0].semilogy(loss_sgd, label='SGD', alpha=0.8)
axes[0].semilogy(loss_mom, label='Momentum (β=0.9)', alpha=0.8)
axes[0].semilogy(loss_nag, label='Nesterov (β=0.9)', alpha=0.8)
axes[0].set_xlabel('迭代次数')
axes[0].set_ylabel('损失值 (log)')
axes[0].set_title('收敛曲线对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 优化轨迹（在二次函数上可视化）
def simple_quad(x):
    return 0.5*(x[0]**2 + 10*x[1]**2)
def simple_quad_grad(x):
    return np.array([x[0], 10*x[1]])

x0_q = np.array([4.0, 3.0])
_, path_q_sgd = sgd(simple_quad_grad, x0_q, lr=0.05, n_iter=80)
_, path_q_mom = momentum(simple_quad_grad, x0_q, lr=0.05, beta=0.9, n_iter=80)

xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
zz = 0.5*(xx**2 + 10*yy**2)
axes[1].contour(xx, yy, zz, levels=20, cmap='viridis', alpha=0.6)
axes[1].plot(path_q_sgd[:,0], path_q_sgd[:,1], 'r-o', markersize=2, linewidth=1, label='SGD')
axes[1].plot(path_q_mom[:,0], path_q_mom[:,1], 'b-s', markersize=2, linewidth=1, label='Momentum')
axes[1].plot(0, 0, 'k*', markersize=15)
axes[1].set_title('优化轨迹对比')
axes[1].legend()

plt.tight_layout()
plt.savefig('momentum_comparison.png', dpi=150)
plt.show()
```


## 五、动量系数的选择


| 动量系数 \( \beta \) | 效果 | 适用场景 |
| --- | --- | --- |
| 0 | 退化为纯SGD | 不推荐 |
| 0.5 | 中等动量 | 某些简单问题 |
| 0.9 | 标准选择 | 大多数深度学习任务 |
| 0.99 | 强动量，历史影响大 | 梯度噪声特别大的场景 |
| 0.999 | 极强动量 | 很少使用（Adam中用作β₂） |


> **Note:** **PyTorch中的使用：**
>
>
> `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`
>
>
> `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)`


## 六、理论分析


### 6.1 动量的等价形式


Momentum可以写成二阶微分方程的形式：


$$
\(\ddot{x} + \gamma \dot{x} + \nabla f(x) = 0\)
$$


这是有阻尼的牛顿第二定律，其中 \( \gamma \) 是阻尼系数。当 \( \gamma \to 0 \)（无阻尼），小球永远振荡；当 \( \gamma \) 适中，小球沿着梯度方向加速并最终收敛。


### 6.2 Nesterov加速的数学证明要点


Nesterov在1983年证明：对于凸光滑函数，动量法的收敛率是 \( O(1/k) \)，而NAG可以达到 \( O(1/k^2) \)。关键在于NAG的前瞻步实际上构成了对目标函数的一个"估计"，使得梯度估计更准确。


> **Important:** **核心结论：**
>
>
> - Momentum：抑制震荡，加速收敛，是深度学习优化器的基础组件
>
>
> - NAG：在Momentum基础上加"纠错"机制，理论收敛率更优
>
>
> - 两者结合自适应学习率（如RMSProp）就得到了Adam——目前最流行的优化器


<!-- Converted from: 02_动量法与Nesterov加速.html -->
