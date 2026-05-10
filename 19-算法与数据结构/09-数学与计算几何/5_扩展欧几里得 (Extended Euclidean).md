# 扩展欧几里得算法 (Extended Euclidean Algorithm)

## 一、概念定义与原理

### 1.1 核心思想

在求 $\gcd(a, b)$ 的同时，找到整数 $x, y$ 使得：

$$ax + by = \gcd(a, b)$$

这是**裴蜀定理**的构造性证明。

### 1.2 推导过程

假设已知 $bx' + (a \bmod b)y' = \gcd(b, a \bmod b)$

因为 $a \bmod b = a - \lfloor a/b \rfloor \cdot b$，代入得：

$$ay' + b(x' - \lfloor a/b \rfloor \cdot y') = \gcd(a, b)$$

递推关系：$x = y'$，$y = x' - \lfloor a/b \rfloor \cdot y'$

**边界条件：** $b = 0$ 时，$\gcd(a, 0) = a$，取 $x = 1, y = 0$。

### 1.3 通解形式

若 $(x_0, y_0)$ 是 $ax + by = c$ 的一组特解，则通解为：

$$x = x_0 + k \cdot \frac{b}{g}, \quad y = y_0 - k \cdot \frac{a}{g} \quad (k \in \mathbb{Z})$$

其中 $g = \gcd(a, b)$。

---

## 二、核心应用

### 2.1 模逆元

当 $\gcd(a, m) = 1$ 时，$ax + my = 1$，即 $ax \equiv 1 \pmod{m}$，$x$ 是 $a$ 关于模 $m$ 的逆元。

### 2.2 线性同余方程

求解 $ax \equiv c \pmod{m}$，等价于 $ax + my = c$。有解条件：$\gcd(a, m) \mid c$。

---

## 三、代码实现

### 3.1 递归实现 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    long long g = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return g;
}

// 求 a 关于模 m 的逆元
long long mod_inverse(long long a, long long m) {
    long long x, y;
    long long g = exgcd(a, m, x, y);
    if (g != 1) return -1;
    return (x % m + m) % m;
}
```

### 3.2 迭代实现 - C++

```cpp
long long exgcd_iter(long long a, long long b, long long &x, long long &y) {
    long long x0 = 1, y0 = 0;
    long long x1 = 0, y1 = 1;
    while (b) {
        long long q = a / b;
        long long t;
        t = a; a = b; b = t % b;
        t = x0; x0 = x1; x1 = t - q * x1;
        t = y0; y0 = y1; y1 = t - q * y1;
    }
    x = x0; y = y0;
    return a;
}
```

### 3.3 Python 实现

```python
def exgcd(a, b):
    """返回 (g, x, y) 使得 ax + by = g"""
    if b == 0:
        return a, 1, 0
    g, x, y = exgcd(b, a % b)
    return g, y, x - (a // b) * y

def mod_inverse(a, m):
    g, x, y = exgcd(a, m)
    if g != 1:
        return None
    return (x % m + m) % m

# 测试
g, x, y = exgcd(12, 8)
print(f"gcd={g}, x={x}, y={y}")  # gcd=4, x=1, y=-1
print(mod_inverse(3, 7))  # 5
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 扩展欧几里得 | $O(\log(\min(a,b)))$ | 与普通GCD相同 |
| 求逆元 | $O(\log m)$ | 一次exgcd |
| 线性同余方程 | $O(\log m)$ | 一次exgcd |

---

## 五、竞赛与面试应用场景

### 5.1 典型应用

1. **求模逆元：** 当模数非质数时，费马小定理不适用，用扩展欧几里得
2. **线性同余方程：** $ax \equiv b \pmod{m}$
3. **中国剩余定理：** 配合CRT使用
4. **丢番图方程：** $ax + by = c$ 的整数解

### 5.2 注意事项

- 结果可能为负数，需要 `(x % m + m) % m`
- 求逆元的前提是 $\gcd(a, m) = 1$
- 当 $a$ 为负数时，先转换为 $a \bmod m$
