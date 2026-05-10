# 拉格朗日乘子法与KKT条件 — 最优化理论笔记


## 一、约束优化问题的数学形式


一般约束优化问题的标准形式为：


$$
\(\min_{x \in \mathbb{R}^n} \quad f(x)\)
                \(\text{s.t.} \quad g_i(x) \le 0, \quad i = 1, \dots, m\)
                \(\quad\quad h_j(x) = 0, \quad j = 1, \dots, p\)
$$


其中：


- \( f(x) \)：目标函数
- \( g_i(x) \le 0 \)：不等式约束（共 \( m \) 个）
- \( h_j(x) = 0 \)：等式约束（共 \( p \) 个）
- 可行域：\( \mathcal{D} = \{x \mid g_i(x) \le 0, \; h_j(x) = 0, \; \forall i, j\} \)


> **Note:** **直觉：**
> 我们要在约束划定的"可行域"内找到目标函数的最小值。无约束时直接令梯度为零；有约束时，最优解处梯度不一定为零，而是受到约束的"拉扯"。


## 二、等式约束：拉格朗日乘子法


### 2.1 问题形式


$$
\(\min f(x) \quad \text{s.t.} \quad h_j(x) = 0, \; j=1,\dots,p\)
$$


### 2.2 拉格朗日函数


构造**拉格朗日函数（Lagrangian）**：


$$
\(\mathcal{L}(x, \nu) = f(x) + \sum_{j=1}^{p} \nu_j h_j(x)\)
$$


其中 \( \nu_j \) 称为**拉格朗日乘子**（Lagrange multiplier）。


### 2.3 最优性条件


若 \( x^* \) 是约束优化问题的局部最优解，则存在 \( \nu^* \) 使得：


$$
\(\nabla_x \mathcal{L}(x^*, \nu^*) = \nabla f(x^*) + \sum_{j=1}^{p} \nu_j^* \nabla h_j(x^*) = 0\)
$$


几何解释：在最优点处，目标函数的梯度方向被约束曲面的法向量完全"抵消"。目标函数的梯度必须落在约束梯度张成的空间中。


> **Example:** **例题：**
> 求 \( f(x,y) = x^2 + y^2 \) 在约束 \( x + y = 4 \) 下的最小值。
>
>
>
>
> **解：**
> 构造拉格朗日函数：
>
>
> \(\mathcal{L}(x, y, \nu) = x^2 + y^2 + \nu(x + y - 4)\)
>
>
>
>
> 求偏导令为零：
>
>
> \(\frac{\partial \mathcal{L}}{\partial x} = 2x + \nu = 0 \Rightarrow x = -\nu/2\)
>
>
> \(\frac{\partial \mathcal{L}}{\partial y} = 2y + \nu = 0 \Rightarrow y = -\nu/2\)
>
>
> \(\frac{\partial \mathcal{L}}{\partial \nu} = x + y - 4 = 0\)
>
>
>
>
> 解得：\( x = y = 2, \; \nu = -4 \)。
>
>
> 最优值：\( f(2,2) = 8 \)。


## 三、KKT条件（Karush-Kuhn-Tucker Conditions）


### 3.1 广义拉格朗日函数


对于含不等式约束和等式约束的一般问题，构造：


$$
\(\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)\)
$$


其中 \( \lambda_i \ge 0 \) 为不等式约束对应的乘子。


### 3.2 KKT五个条件


若 \( x^* \) 是约束优化问题的最优解，且满足一定的正则性条件，则存在 \( \lambda^*, \nu^* \) 使得以下五个条件同时成立：


| 条件 | 数学表达 | 含义 |
| --- | --- | --- |
| **①平稳性条件** | \(\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = 0\) | 拉格朗日函数对 \( x \) 的梯度为零 |
| **②原始可行性** | \(g_i(x^*) \le 0, \; h_j(x^*) = 0\) | 最优解满足所有约束 |
| **③对偶可行性** | \(\lambda_i^* \ge 0\) | 不等式约束乘子非负 |
| **④互补松弛性** | \(\lambda_i^* g_i(x^*) = 0\) | 对每个 \( i \)，要么约束不起作用（\( \lambda_i^*=0 \)），要么约束取等号（\( g_i(x^*)=0 \)） |
| **⑤约束规范** | LICQ/MFCQ等 | 保证KKT条件成立的正则性条件 |


> **Important:** **互补松弛性详解：**
> 这是KKT条件中最精妙的部分。
>
>
> - 若 \( \lambda_i^* > 0 \)，则 \( g_i(x^*) = 0 \)：约束"绑定"（active），严格影响最优解
>
>
> - 若 \( g_i(x^*) < 0 \)，则 \( \lambda_i^* = 0 \)：约束"松"（inactive），不影响最优解
>
>
> 这告诉我们哪些约束真正"起作用"了。


### 3.3 KKT条件的必要性与充分性


- **必要性：**
   若问题满足约束规范（如LICQ），则局部最优解必满足KKT条件
- **充分性：**
   若问题是凸优化问题（\( f \) 凸、\( g_i \) 凸、\( h_j \) 仿射），则满足KKT条件的点一定是全局最优解
- **充要性：**
   对于凸优化问题，KKT条件是全局最优解的充要条件


## 四、KKT条件例题


> **Example:** **例题：**
> 求解以下约束优化问题：
>
>
> \(\min \quad f(x,y) = (x-1)^2 + (y-2)^2\)
>
>
> \(\text{s.t.} \quad g_1(x,y) = x + y - 3 \le 0\)
>
>
> \(\quad\quad\quad g_2(x,y) = -x \le 0\)（即 \( x \ge 0 \)）
>
>
> \(\quad\quad\quad g_3(x,y) = -y \le 0\)（即 \( y \ge 0 \)）
>
>
>
>
> **解：**
> 无约束最小值在 \( (1,2) \)，但 \( g_1(1,2) = 0 \)，恰好在边界上。
>
>
>
>
> 构造拉格朗日函数：
>
>
> \(\mathcal{L} = (x-1)^2 + (y-2)^2 + \lambda_1(x+y-3) + \lambda_2(-x) + \lambda_3(-y)\)
>
>
>
>
> KKT条件：
>
>
> \(\frac{\partial \mathcal{L}}{\partial x} = 2(x-1) + \lambda_1 - \lambda_2 = 0\)
>
>
> \(\frac{\partial \mathcal{L}}{\partial y} = 2(y-2) + \lambda_1 - \lambda_3 = 0\)
>
>
> 互补松弛：\(\lambda_1(x+y-3)=0, \; \lambda_2(-x)=0, \; \lambda_3(-y)=0\)
>
>
> \(\lambda_i \ge 0\)
>
>
>
>
> 尝试 \( \lambda_2=\lambda_3=0, \lambda_1>0 \Rightarrow x+y=3 \)：
>
>
> \(2(x-1)+\lambda_1=0 \Rightarrow \lambda_1 = -2(x-1)\)
>
>
> \(2(y-2)+\lambda_1=0 \Rightarrow \lambda_1 = -2(y-2)\)
>
>
> 所以 \( x-1 = y-2 \Rightarrow x=y-1 \)，结合 \( x+y=3 \)：\( x=1, y=2, \lambda_1=0 \)。
>
>
>
>
> 答案：最优解 \( (x^*, y^*) = (1, 2) \)，最优值 \( f^* = 0 \)。


## 五、Python代码：scipy求解约束优化


```
import numpy as np
from scipy.optimize import minimize

# ==========================================
# 示例1: 等式约束优化
# min x^2 + y^2  s.t. x + y = 4
# ==========================================
def objective(x):
    return x[0]**2 + x[1]**2

def eq_constraint(x):
    return x[0] + x[1] - 4  # = 0

constraints = [{'type': 'eq', 'fun': eq_constraint}]
x0 = np.array([1.0, 1.0])

result = minimize(objective, x0, constraints=constraints, method='SLSQP')
print("=== 等式约束优化 ===")
print(f"最优解: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
print(f"最优值: {result.fun:.4f}")
print(f"拉格朗日乘子 (等式): {result.vjac if hasattr(result, 'vjac') else 'N/A'}")
print()

# ==========================================
# 示例2: 不等式约束优化
# min (x-1)^2 + (y-2)^2  s.t. x+y <= 3, x>=0, y>=0
# ==========================================
def objective2(x):
    return (x[0]-1)**2 + (x[1]-2)**2

constraints2 = [
    {'type': 'ineq', 'fun': lambda x: 3 - x[0] - x[1]},   # x+y <= 3
    {'type': 'ineq', 'fun': lambda x: x[0]},                 # x >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]},                 # y >= 0
]

result2 = minimize(objective2, x0, constraints=constraints2, method='SLSQP')
print("=== 不等式约束优化 ===")
print(f"最优解: x={result2.x[0]:.4f}, y={result2.x[1]:.4f}")
print(f"最优值: {result2.fun:.4f}")
print()

# ==========================================
# 示例3: 带KKT乘子分析的完整示例
# min x1^2 + x2^2 + x3^2  s.t. x1+x2+x3=1, x1,x2,x3 >= 0
# ==========================================
def objective3(x):
    return np.sum(x**2)

eq_cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = [(0, None), (0, None), (0, None)]

result3 = minimize(objective3, [0.3, 0.3, 0.4],
                   constraints=eq_cons, bounds=bounds, method='SLSQP')
print("=== 投影到概率单纯形 ===")
print(f"最优解: {result3.x}")
print(f"最优值: {result3.fun:.6f}")
print(f"约束验证: sum(x)={np.sum(result3.x):.6f}")
# 理论解: x1=x2=x3=1/3, f*=1/3
```


## 六、KKT条件的工程意义


- **SVM中的应用：**
   SVM的对偶问题就是通过KKT条件推导的。支持向量恰好对应互补松弛性中 \( \lambda_i > 0 \) 的样本
- **神经网络剪枝：**
   利用互补松弛性判断哪些权重可以被剪掉
- **资源分配：**
   经济学中影子价格（Shadow Price）本质上就是拉格朗日乘子
- **正则化解释：**
   L2正则化可以等价转化为带范数约束的优化问题，KKT条件建立了两者联系


> **Note:** **总结：**
> 拉格朗日乘子法将约束优化转化为无约束问题的思路是优化理论的核心技巧。KKT条件给出了最优解的完整刻画，是理论分析和算法设计的基础工具。对于凸优化问题，KKT条件既是必要的也是充分的。


<!-- Converted from: 02_拉格朗日乘子法与KKT条件.html -->
