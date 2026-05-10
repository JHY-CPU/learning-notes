# 中国剩余定理 (Chinese Remainder Theorem, CRT)

## 一、概念定义与原理

### 1.1 历史背景

中国剩余定理最早见于南北朝时期的《孙子算经》，其中有"物不知数"问题：

> 今有物不知其数，三三数之剩二，五五数之剩三，七七数之剩二，问物几何？

即求满足 $x \equiv 2 \pmod{3}$, $x \equiv 3 \pmod{5}$, $x \equiv 2 \pmod{7}$ 的最小正整数。

### 1.2 定理表述

设 $m_1, m_2, \ldots, m_k$ 为**两两互质**的正整数，则同余方程组：

$$\begin{cases} x \equiv a_1 \pmod{m_1} \\ x \equiv a_2 \pmod{m_2} \\ \vdots \\ x \equiv a_k \pmod{m_k} \end{cases}$$

在模 $M = m_1 m_2 \cdots m_k$ 下有唯一解。

### 1.3 构造解

令 $M = \prod_{i=1}^{k} m_i$，$M_i = M / m_i$，$t_i$ 为 $M_i$ 关于模 $m_i$ 的逆元，则：

$$x = \sum_{i=1}^{k} a_i M_i t_i \pmod{M}$$

---

## 二、扩展中国剩余定理 (EXCRT)

当模数**不互质**时，两两合并方程。

给定 $x \equiv a_1 \pmod{m_1}$ 和 $x \equiv a_2 \pmod{m_2}$，令 $x = a_1 + k \cdot m_1$，代入得：

$$k \cdot m_1 \equiv a_2 - a_1 \pmod{m_2}$$

令 $g = \gcd(m_1, m_2)$，方程有解当且仅当 $g \mid (a_2 - a_1)$。合并为：

$$x \equiv x_0 \pmod{\text{lcm}(m_1, m_2)}$$

---

## 三、代码实现

### 3.1 基本CRT - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long power(long long a, long long b, long long m) {
    long long r = 1; a %= m;
    while (b) { if (b & 1) r = r * a % m; a = a * a % m; b >>= 1; }
    return r;
}

long long CRT(long long a[], long long m[], int n) {
    long long M = 1;
    for (int i = 0; i < n; i++) M *= m[i];
    long long result = 0;
    for (int i = 0; i < n; i++) {
        long long Mi = M / m[i];
        long long ti = power(Mi, m[i] - 2, m[i]);
        result = (result + a[i] % M * Mi % M * ti % M) % M;
    }
    return (result % M + M) % M;
}
```

### 3.2 扩展CRT - C++

```cpp
long long exgcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    long long g = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return g;
}

long long EXCRT(long long a[], long long m[], int n) {
    long long a1 = a[0], m1 = m[0];
    for (int i = 1; i < n; i++) {
        long long a2 = a[i], m2 = m[i];
        long long g = __gcd(m1, m2);
        if ((a2 - a1) % g != 0) return -1;
        long long x, y;
        exgcd(m1 / g, m2 / g, x, y);
        long long lcm = m1 / g * m2;
        long long k = ((a2 - a1) / g % (m2 / g) * x % (m2 / g) + (m2 / g)) % (m2 / g);
        a1 = a1 + k * m1;
        m1 = lcm;
        a1 = (a1 % m1 + m1) % m1;
    }
    return a1;
}
```

### 3.3 Python 实现

```python
def CRT(a, m):
    M = 1
    for mi in m: M *= mi
    result = 0
    for ai, mi in zip(a, m):
        Mi = M // mi
        ti = pow(Mi, mi - 2, mi)
        result = (result + ai * Mi * ti) % M
    return result

def EXCRT(a, m):
    a1, m1 = a[0], m[0]
    for i in range(1, len(a)):
        a2, m2 = a[i], m[i]
        g, x, y = exgcd(m1, m2)
        if (a2 - a1) % g != 0: return -1
        lcm = m1 // g * m2
        k = (a2 - a1) // g * x % (m2 // g)
        a1 = a1 + k * m1
        m1 = lcm
        a1 = (a1 % m1 + m1) % m1
    return a1

print(CRT([2, 3, 2], [3, 5, 7]))  # 23
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 适用条件 |
|------|-----------|---------|
| 基本CRT | $O(k \log M)$ | 模数两两互质 |
| 扩展CRT | $O(k \log(\max m_i))$ | 模数任意 |

---

## 五、竞赛与面试应用场景

### 5.1 常见题型

1. **大数取模：** 将大数分解为多个小模数下的表示，再用CRT合并
2. **循环节问题：** 多个周期性事件同时发生的最小时间
3. **编码与校验：** 冗余数系统

### 5.2 注意事项

- 基本CRT要求模数**两两互质**
- 扩展CRT中判断无解的条件：$\gcd(m_1, m_2) \nmid (a_2 - a_1)$
- 注意中间结果的溢出问题，必要时使用 `__int128`
