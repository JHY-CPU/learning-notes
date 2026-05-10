# 线性规划 (Linear Programming)

## 一、概念定义与原理

### 1.1 基本形式

线性规划的标准形式：

**最大化（或最小化）：** $c^T x$

**约束条件：**
$$Ax \leq b, \quad x \geq 0$$

其中 $x$ 是决策变量向量，$c$ 是目标函数系数，$A$ 是约束矩阵，$b$ 是约束右端。

### 1.2 几何解释

可行域是多面体（由半平面交形成），最优解一定在顶点上取得。

### 1.3 对偶性

原问题（最大化 $c^T x$，$Ax \leq b$，$x \geq 0$）的对偶是最小化 $b^T y$，$A^T y \geq c$，$y \geq 0$。

**强对偶定理：** 若原问题有最优解，则对偶问题也有最优解，且最优值相等。

---

## 二、核心算法

### 2.1 单纯形法

**基本步骤：**
1. 将问题转化为标准形式（引入松弛变量）
2. 从一个基本可行解开始
3. 选择一个非基变量入基（改善目标函数）
4. 选择一个基变量出基（保持可行性）
5. 重复直到无法改善

### 2.2 二维线性规划

对于二维问题，可以用**半平面交**求可行域，然后在顶点上检查目标函数值。时间复杂度 $O(n \log n)$。

---

## 三、代码实现

### 3.1 单纯形法 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 单纯形法求解线性规划
// 最大化 c^T x，约束 Ax <= b, x >= 0
const double INF = 1e18;
const double EPS = 1e-9;

// A: m x n 矩阵 (m个约束, n个变量)
// b: m维向量 (约束右端)
// c: n维向量 (目标函数系数)
// 返回最优值，x 存储最优解
double simplex(vector<vector<double>>& A, vector<double>& b, vector<double>& c,
               vector<double>& x) {
    int m = A.size(), n = c.size();
    // 构造单纯形表
    // 表的行: m+1 (m个约束+1个目标函数)
    // 表的列: n+m+1 (n个变量+m个松弛变量+1个右端)
    vector<vector<double>> T(m + 1, vector<double>(n + m + 1, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) T[i][j] = A[i][j];
        T[i][n + i] = 1; // 松弛变量
        T[i][n + m] = b[i];
    }
    for (int j = 0; j < n; j++) T[m][j] = -c[j];

    while (true) {
        // 找入基变量（最负的系数）
        int pivot_col = -1;
        for (int j = 0; j < n + m; j++) {
            if (T[m][j] < -EPS && (pivot_col == -1 || T[m][j] < T[m][pivot_col]))
                pivot_col = j;
        }
        if (pivot_col == -1) break; // 最优

        // 找出基变量（最小比值）
        int pivot_row = -1;
        double min_ratio = INF;
        for (int i = 0; i < m; i++) {
            if (T[i][pivot_col] > EPS) {
                double ratio = T[i][n + m] / T[i][pivot_col];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    pivot_row = i;
                }
            }
        }
        if (pivot_row == -1) return INF; // 无界

        // 高斯消元
        double pivot = T[pivot_row][pivot_col];
        for (int j = 0; j <= n + m; j++) T[pivot_row][j] /= pivot;
        for (int i = 0; i <= m; i++) {
            if (i == pivot_row) continue;
            double factor = T[i][pivot_col];
            for (int j = 0; j <= n + m; j++) T[i][j] -= factor * T[pivot_row][j];
        }
    }

    x.assign(n, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(T[i][j] - 1) < EPS) {
                bool is_basic = true;
                for (int k = 0; k < m; k++) {
                    if (k != i && abs(T[k][j]) > EPS) { is_basic = false; break; }
                }
                if (is_basic) { x[j] = T[i][n + m]; break; }
            }
        }
    }
    return T[m][n + m];
}
```

### 3.2 Python 实现（二维线性规划）

```python
def linear_program_2d(constraints, objective):
    """二维线性规划：constraints = [(a,b,c)] 表示 ax+by<=c
    objective = (cx, cy) 最大化 cx*x + cy*y"""
    from itertools import combinations
    best = None
    best_val = float('-inf')
    # 枚举所有约束的交点
    n = len(constraints)
    for i in range(n):
        for j in range(i+1, n):
            a1, b1, c1 = constraints[i]
            a2, b2, c2 = constraints[j]
            det = a1*b2 - a2*b1
            if abs(det) < 1e-12: continue
            x = (c1*b2 - c2*b1) / det
            y = (a1*c2 - a2*c1) / det
            # 检查是否满足所有约束
            feasible = True
            for a, b, c in constraints:
                if a*x + b*y > c + 1e-9:
                    feasible = False
                    break
            if feasible:
                val = objective[0]*x + objective[1]*y
                if val > best_val:
                    best_val = val
                    best = (x, y)
    return best, best_val

# 测试
constraints = [(1, 0, 4), (0, 1, 6), (1, 1, 8), (-1, 0, 0), (0, -1, 0)]
print(linear_program_2d(constraints, (3, 5)))  # 最大化 3x+5y
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 说明 |
|------|-----------|------|
| 单纯形法 | 指数最坏，实际多项式 | 工程高效 |
| 椭球法 | $O(n^4 L)$ | 理论多项式 |
| 内点法 | $O(n^{3.5} L)$ | 大规模问题 |
| 二维LP | $O(n \log n)$ | 半平面交 |

---

## 五、竞赛与面试应用场景

1. **资源分配：** 在约束下最大化利润/最小化成本
2. **二维LP：** 配合半平面交
3. **对偶技巧：** 将某些组合优化问题转化为LP
4. **费用流：** 网络流与线性规划的关系
