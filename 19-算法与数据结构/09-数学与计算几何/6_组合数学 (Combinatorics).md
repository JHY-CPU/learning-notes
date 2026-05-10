# 组合数学 (Combinatorics)

## 一、概念定义与原理

### 1.1 排列与组合

**排列：** 从 $n$ 个不同元素中取 $m$ 个按顺序排列：

$$A_n^m = P(n, m) = \frac{n!}{(n-m)!}$$

**组合：** 从 $n$ 个不同元素中取 $m$ 个不考虑顺序：

$$C_n^m = \binom{n}{m} = \frac{n!}{m!(n-m)!}$$

### 1.2 二项式定理

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$$

特别地：$(1+1)^n = \sum_{k=0}^{n} \binom{n}{k} = 2^n$

### 1.3 组合数性质

1. **对称性：** $\binom{n}{m} = \binom{n}{n-m}$
2. **Pascal公式：** $\binom{n}{m} = \binom{n-1}{m} + \binom{n-1}{m-1}$
3. **吸收公式：** $m \binom{n}{m} = n \binom{n-1}{m-1}$
4. **范德蒙恒等式：** $\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k}\binom{n}{r-k}$

---

## 二、Lucas 定理

设 $p$ 为质数，将 $n, m$ 写成 $p$ 进制：

$$\binom{n}{m} \equiv \prod_{i=0}^{k} \binom{n_i}{m_i} \pmod{p}$$

适用场景：$n, m$ 很大但模数 $p$ 较小且为质数。

---

## 三、代码实现

### 3.1 预处理阶乘求组合数 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 2e6 + 5;
const int MOD = 1e9 + 7;
long long fac[MAXN], inv_fac[MAXN];

long long power(long long a, long long b, long long m) {
    long long r = 1; a %= m;
    while (b) { if (b & 1) r = r * a % m; a = a * a % m; b >>= 1; }
    return r;
}

void precompute() {
    fac[0] = 1;
    for (int i = 1; i < MAXN; i++) fac[i] = fac[i-1] * i % MOD;
    inv_fac[MAXN-1] = power(fac[MAXN-1], MOD-2, MOD);
    for (int i = MAXN-2; i >= 0; i--) inv_fac[i] = inv_fac[i+1] * (i+1) % MOD;
}

long long C(int n, int m) {
    if (m < 0 || m > n) return 0;
    return fac[n] % MOD * inv_fac[m] % MOD * inv_fac[n-m] % MOD;
}
```

### 3.2 Lucas 定理 - C++

```cpp
long long lucas(long long n, long long m, long long p) {
    if (m == 0) return 1;
    return lucas(n / p, m / p, p) % p * C(n % p, m % p) % p;
}
```

### 3.3 Python 实现

```python
MOD = 10**9 + 7

def power(a, b, m):
    r = 1; a %= m
    while b:
        if b & 1: r = r * a % m
        a = a * a % m; b >>= 1
    return r

def precompute(n):
    fac = [1] * (n + 1)
    for i in range(1, n + 1): fac[i] = fac[i-1] * i % MOD
    inv_fac = [1] * (n + 1)
    inv_fac[n] = power(fac[n], MOD - 2, MOD)
    for i in range(n - 1, -1, -1): inv_fac[i] = inv_fac[i+1] * (i + 1) % MOD
    return fac, inv_fac

def C(n, m, fac, inv_fac):
    if m < 0 or m > n: return 0
    return fac[n] * inv_fac[m] % MOD * inv_fac[n-m] % MOD

fac, inv_fac = precompute(100)
print(C(10, 3, fac, inv_fac))  # 120
```

---

## 四、常见组合数学模型

### 4.1 隔板法

将 $n$ 个相同的球放入 $m$ 个不同的盒子：
- 每盒非空：$C_{n-1}^{m-1}$
- 允许空盒：$C_{n+m-1}^{m-1}$

### 4.2 卡特兰数

$$C_n = \frac{1}{n+1}\binom{2n}{n} = \binom{2n}{n} - \binom{2n}{n-1}$$

递推：$C_0 = 1$，$C_n = \sum_{i=0}^{n-1} C_i C_{n-1-i}$

应用：合法括号序列数、二叉树个数、出栈序列数。

### 4.3 斯特林数

**第二类斯特林数** $S(n, k)$：将 $n$ 个不同元素分成 $k$ 个非空集合的方式数。

$$S(n, k) = k \cdot S(n-1, k) + S(n-1, k-1)$$

---

## 五、复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| Pascal 三角 | $O(n^2)$ | $O(n^2)$ | $n \leq 1000$ |
| 预处理阶乘 | $O(N + \log MOD)$ | $O(N)$ | $n \leq 2 \times 10^6$ |
| Lucas 定理 | $O(p \log_p n)$ | $O(p)$ | $n$ 大，$p$ 小质数 |
