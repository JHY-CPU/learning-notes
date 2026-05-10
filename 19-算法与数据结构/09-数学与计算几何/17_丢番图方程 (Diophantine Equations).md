# 丢番图方程 (Diophantine Equations)

## 一、概念定义与原理

### 1.1 定义

**丢番图方程**是指系数为整数、要求**整数解**的方程。以古希腊数学家丢番图命名。

### 1.2 线性丢番图方程

**形式：** $ax + by = c$，其中 $a, b, c$ 为给定整数，求整数解 $(x, y)$。

**有解条件：** 方程有整数解当且仅当 $\gcd(a, b) \mid c$。

### 1.3 通解形式

若 $(x_0, y_0)$ 是一组特解，$g = \gcd(a, b)$，则通解为：

$$x = x_0 + k \cdot \frac{b}{g}, \quad y = y_0 - k \cdot \frac{a}{g} \quad (k \in \mathbb{Z})$$

---

## 二、核心算法

### 2.1 求解步骤

1. 用**扩展欧几里得**求 $ax_0 + by_0 = g$
2. 若 $c \bmod g \neq 0$，无解
3. 特解：$x' = x_0 \cdot (c/g)$，$y' = y_0 \cdot (c/g)$
4. 通解：$x = x' + k \cdot (b/g)$，$y = y' - k \cdot (a/g)$

### 2.2 最小非负解

对于 $x \geq 0, y \geq 0$ 的约束，确定 $k$ 的范围：

$$k \geq \lceil -x' \cdot g / b \rceil, \quad k \leq \lfloor y' \cdot g / a \rfloor$$

### 2.3 佩尔方程

$x^2 - ny^2 = 1$（$n$ 为非平方正整数）

有无穷多组正整数解，最小解可用**连分数**求得，其余解由最小解的幂生成。

---

## 三、代码实现

### 3.1 线性丢番图方程 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    long long g = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return g;
}

// 求 ax + by = c 的最小非负解
// 返回 {x, y}，无解返回 {-1, -1}
pair<long long, long long> diophantine(long long a, long long b, long long c) {
    long long x, y;
    long long g = exgcd(a, b, x, y);
    if (c % g != 0) return {-1, -1};
    x *= c / g; y *= c / g;
    // 调整到最小非负解
    long long step_x = b / g, step_y = a / g;
    // x + k*step_x >= 0 的最小 k
    if (step_x < 0) step_x = -step_x;
    if (step_y < 0) step_y = -step_y;
    // 使 x 为最小非负
    long long k = x / step_x;
    if (x < 0) k--;
    x -= k * step_x; y += k * step_y;
    return {x, y};
}
```

### 3.2 统计非负解个数

```cpp
// ax + by = c 的非负整数解个数
long long count_solutions(long long a, long long b, long long c) {
    long long x, y;
    long long g = exgcd(a, b, x, y);
    if (c % g != 0) return 0;
    x *= c / g; y *= c / g;
    long long step_x = b / g, step_y = a / g;
    if (step_x < 0) step_x = -step_x;
    if (step_y < 0) step_y = -step_y;
    // 通解 x = x0 + k*step_x, y = y0 - k*(a/g)
    // x >= 0: k >= -x0/step_x
    // y >= 0: k <= y0/(a/g)
    // 需要仔细计算边界...
    long long k_min = (x < 0) ? (-x + step_x - 1) / step_x : 0;
    x += k_min * step_x; y -= k_min * (a / g);
    if (y < 0) return 0;
    long long k_max = y / (a / g);
    return k_max + 1;
}
```

### 3.3 Python 实现

```python
def exgcd(a, b):
    if b == 0: return a, 1, 0
    g, x, y = exgcd(b, a % b)
    return g, y, x - (a // b) * y

def diophantine(a, b, c):
    """求 ax + by = c 的一组非负整数解"""
    g, x, y = exgcd(a, b)
    if c % g != 0: return None
    x *= c // g; y *= c // g
    step = b // g
    # 调整 x 到最小非负
    if step != 0:
        k = x // abs(step) if step > 0 else -x // abs(step)
        x -= k * step; y += k * (a // g)
    # 确保非负
    if x < 0:
        k = (-x + abs(step) - 1) // abs(step)
        x += k * abs(step); y -= k * (a // g)
    return (x, y) if x >= 0 and y >= 0 else None

print(diophantine(3, 6, 9))  # (3, 0) 或其他非负解
```

### 3.4 佩尔方程

```cpp
// 求 x^2 - n*y^2 = 1 的最小正整数解
// 用连分数展开
pair<long long, long long> pell_equation(long long n) {
    long long a0 = (long long)sqrt(n);
    if (a0 * a0 == n) return {-1, -1}; // n是完全平方，无解
    // 连分数展开，找循环节
    long long m = 0, d = 1, a = a0;
    long long p_prev = 1, q_prev = 0;
    long long p_curr = a0, q_curr = 1;
    while (true) {
        m = d * a - m;
        d = (n - m * m) / d;
        a = (a0 + m) / d;
        long long p_next = a * p_curr + p_prev;
        long long q_next = a * q_curr + q_prev;
        if (p_next * p_next - n * q_next * q_next == 1) {
            return {p_next, q_next};
        }
        p_prev = p_curr; q_prev = q_curr;
        p_curr = p_next; q_curr = q_next;
    }
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 求解线性丢番图 | $O(\log(\min(a,b)))$ | 一次exgcd |
| 统计解个数 | $O(\log(\min(a,b)))$ | 一次exgcd |
| 佩尔方程 | $O(\sqrt{n} \log n)$ | 连分数 |

---

## 五、竞赛与面试应用场景

1. **线性方程整数解：** $ax + by = c$
2. **不定方程计数：** 非负整数解个数
3. **CRT相关：** 扩展CRT中的方程合并
4. **佩尔方程：** 某些数论题目
