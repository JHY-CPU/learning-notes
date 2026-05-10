# 数值算法 (Numerical Algorithms)

## 一、概念定义与原理

数值算法用于在计算机上近似求解数学问题。本节介绍二分求根、牛顿法和数值积分三种常用方法。

### 1.1 误差与精度

- **绝对误差：** $|x_{\text{approx}} - x_{\text{exact}}|$
- **相对误差：** $|x_{\text{approx}} - x_{\text{exact}}| / |x_{\text{exact}}|$
- 竞赛中通常要求绝对误差 $\leq 10^{-6}$ 或 $10^{-9}$

---

## 二、核心算法

### 2.1 二分法求根

若 $f(x)$ 在 $[a, b]$ 上连续且 $f(a) \cdot f(b) < 0$，则 $[a, b]$ 内存在零点。

每次取中点 $m = (a+b)/2$，根据 $f(m)$ 的符号缩小范围。

收敛速度：线性收敛，每次减少一位有效数字。

### 2.2 牛顿法（Newton's Method）

给定 $f(x)$ 和初始猜测 $x_0$，迭代：

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

收敛速度：二次收敛（误差平方递减），但需要 $f'(x) \neq 0$ 且初始值足够接近根。

### 2.3 数值积分

**梯形法则：** $\int_a^b f(x)dx \approx \frac{b-a}{2}(f(a) + f(b))$

**Simpson 法则：** $\int_a^b f(x)dx \approx \frac{b-a}{6}(f(a) + 4f(\frac{a+b}{2}) + f(b))$

**自适应 Simpson：** 递归地在误差大的区间细分。

---

## 三、代码实现

### 3.1 二分求根 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 二分求 f(x) = 0 在 [a, b] 上的根
double bisection(function<double(double)> f, double a, double b, double eps = 1e-9) {
    while (b - a > eps) {
        double m = (a + b) / 2;
        if (f(a) * f(m) <= 0) b = m;
        else a = m;
    }
    return (a + b) / 2;
}
```

### 3.2 牛顿法 - C++

```cpp
double newton(function<double(double)> f, function<double(double)> df,
              double x0, double eps = 1e-9, int max_iter = 100) {
    for (int i = 0; i < max_iter; i++) {
        double fx = f(x0);
        double dfx = df(x0);
        if (abs(dfx) < 1e-15) break;
        double x1 = x0 - fx / dfx;
        if (abs(x1 - x0) < eps) return x1;
        x0 = x1;
    }
    return x0;
}
```

### 3.3 自适应 Simpson 积分 - C++

```cpp
double simpson(function<double(double)> f, double a, double b) {
    return (b - a) / 6 * (f(a) + 4 * f((a+b)/2) + f(b));
}

double adaptive_simpson(function<double(double)> f, double a, double b, double eps) {
    double m = (a + b) / 2;
    double S = simpson(f, a, b);
    double S1 = simpson(f, a, m);
    double S2 = simpson(f, m, b);
    if (abs(S1 + S2 - S) <= 15 * eps) return S1 + S2 + (S1 + S2 - S) / 15;
    return adaptive_simpson(f, a, m, eps/2) + adaptive_simpson(f, m, b, eps/2);
}
```

### 3.4 Python 实现

```python
import math

def bisection(f, a, b, eps=1e-9):
    """二分求根"""
    while b - a > eps:
        m = (a + b) / 2
        if f(a) * f(m) <= 0: b = m
        else: a = m
    return (a + b) / 2

def newton(f, df, x0, eps=1e-9):
    """牛顿法"""
    for _ in range(100):
        fx = f(x0); dfx = df(x0)
        if abs(dfx) < 1e-15: break
        x1 = x0 - fx / dfx
        if abs(x1 - x0) < eps: return x1
        x0 = x1
    return x0

def simpson_integrate(f, a, b):
    """Simpson 法则"""
    return (b - a) / 6 * (f(a) + 4 * f((a+b)/2) + f(b))

def adaptive_simpson(f, a, b, eps=1e-9):
    """自适应 Simpson 积分"""
    m = (a + b) / 2
    S = simpson_integrate(f, a, b)
    S1 = simpson_integrate(f, a, m)
    S2 = simpson_integrate(f, m, b)
    if abs(S1 + S2 - S) <= 15 * eps:
        return S1 + S2 + (S1 + S2 - S) / 15
    return adaptive_simpson(f, a, m, eps/2) + adaptive_simpson(f, m, b, eps/2)

# 测试
print(bisection(lambda x: x**2 - 2, 1, 2))     # 约 1.41421
print(newton(lambda x: x**2 - 2, lambda x: 2*x, 1.5))  # 约 1.41421
print(adaptive_simpson(math.sin, 0, math.pi))   # 约 2.0
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 收敛速度 |
|------|-----------|---------|
| 二分法 | $O(\log(\frac{b-a}{\epsilon}))$ | 线性 |
| 牛顿法 | $O(\log(\log(1/\epsilon)))$ | 二次 |
| Simpson | $O(n)$ | 四次（复合） |
| 自适应Simpson | 取决于函数 | 自动精度 |

---

## 五、竞赛与面试应用场景

1. **方程求根：** 解非线性方程
2. **数值积分：** 计算无解析解的定积分
3. **最优化：** 三分法求单峰函数极值
4. **浮点二分：** 竞赛中常见的精度处理技巧

**三分法求极值：**
```cpp
double ternary_search(function<double(double)> f, double lo, double hi) {
    for (int i = 0; i < 200; i++) {
        double m1 = lo + (hi - lo) / 3;
        double m2 = hi - (hi - lo) / 3;
        if (f(m1) < f(m2)) lo = m1;  // 求最小值
        else hi = m2;
    }
    return (lo + hi) / 2;
}
```
