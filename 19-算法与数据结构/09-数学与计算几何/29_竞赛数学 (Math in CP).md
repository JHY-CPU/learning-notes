# 竞赛数学 (Math in CP)

## 一、竞赛中常用数学知识点

### 1.1 必备公式

**等差数列：** $S_n = \frac{n(a_1 + a_n)}{2} = na_1 + \frac{n(n-1)}{2}d$

**等比数列：** $S_n = \frac{a_1(1-q^n)}{1-q}$（$q \neq 1$）

**调和级数：** $H_n = \sum_{i=1}^{n} \frac{1}{i} \approx \ln n + \gamma$（$\gamma \approx 0.5772$）

**幂和公式：** $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$，$\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$

### 1.2 常用恒等式

- $(a+b)^2 = a^2 + 2ab + b^2$
- $a^3 + b^3 = (a+b)(a^2 - ab + b^2)$
- $a^n - 1 = (a-1)(a^{n-1} + a^{n-2} + \cdots + 1)$
- $\gcd(a,b) \cdot \text{lcm}(a,b) = a \cdot b$

### 1.3 模运算规则

- $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$
- $(a \cdot b) \bmod m = ((a \bmod m) \cdot (b \bmod m)) \bmod m$
- $(a - b) \bmod m = ((a \bmod m) - (b \bmod m) + m) \bmod m$
- 除法取模：$a/b \bmod m = a \cdot b^{-1} \bmod m$

---

## 二、竞赛常用技巧

### 2.1 整数溢出处理

```cpp
// 安全乘法（防 long long 溢出）
long long safe_mul(long long a, long long b, long long m) {
    return (__int128)a * b % m;
}

// 或者用二进制分解
long long mul_mod(long long a, long long b, long long m) {
    long long r = 0;
    a %= m;
    while (b) {
        if (b & 1) r = (r + a) % m;
        a = (a + a) % m;
        b >>= 1;
    }
    return r;
}
```

### 2.2 浮点精度处理

```cpp
const double EPS = 1e-9;
bool eq(double a, double b) { return abs(a - b) < EPS; }
bool lt(double a, double b) { return a - b < -EPS; }
bool gt(double a, double b) { return a - b > EPS; }

// 向上取整（安全版）
long long safe_ceil(long long a, long long b) {
    return (a + b - 1) / b;  // a > 0, b > 0
}
```

### 2.3 二进制技巧

```cpp
int popcount(int x) { return __builtin_popcount(x); }  // 1的个数
int ctz(int x) { return __builtin_ctz(x); }            // 尾部0的个数
int clz(int x) { return __builtin_clz(x); }            // 前导0的个数
int lowbit(int x) { return x & (-x); }                 // 最低位1
bool is_pow2(int x) { return x && !(x & (x - 1)); }    // 是否为2的幂
```

---

## 三、常见数学模型

### 3.1 期望DP

```cpp
// 掷硬币直到出现连续k次正面，期望次数
// E(k) = 2 * E(k-1) + 2, E(1) = 2
long long expected_coins(int k) {
    // E(k) = 2^(k+1) - 2
    return (1LL << (k + 1)) - 2;
}
```

### 3.2 递推加速

```cpp
// 对于线性递推 f(n) = a*f(n-1) + b*f(n-2)
// 用矩阵快速幂加速
// [f(n)]   [a b]^n-1   [f(1)]
// [f(n-1)] [1 0]    *  [f(0)]
```

### 3.3 周期性

```cpp
// Fibonacci 数列 mod m 的 Pisano 周期
long long pisano(long long m) {
    long long a = 0, b = 1;
    for (long long i = 1; i <= 6 * m; i++) {
        long long c = (a + b) % m;
        a = b; b = c;
        if (a == 0 && b == 1) return i;
    }
    return -1;
}
```

---

## 四、Python 大数运算

```python
# Python 原生大数支持，不需要担心溢出
print(2 ** 1000)          # 直接计算大数
print(pow(2, 1000, 10**9+7))  # 快速幂取模

# 组合数
from math import comb
print(comb(100, 50))      # Python 3.8+

# 最大公约数
from math import gcd
print(gcd(12, 18))        # 6
```

---

## 五、竞赛经验总结

### 5.1 常见卡点

1. **$n = 0$ 或 $n = 1$ 的边界情况**
2. **负数取模：** `(-a) % m` 在不同语言中行为不同
3. **整数除法方向：** C++ 中 `-3/2 = -1`，Python 中 `-3//2 = -2`
4. **大数溢出：** 两数相乘前检查是否需要 `__int128`

### 5.2 时间复杂度估算

| 数据规模 | 适用算法 |
|---------|---------|
| $n \leq 10$ | $O(n!)$ 暴力 |
| $n \leq 20$ | $O(2^n)$ 状压/容斥 |
| $n \leq 50$ | $O(n^4)$ |
| $n \leq 500$ | $O(n^3)$ DP/高斯消元 |
| $n \leq 5000$ | $O(n^2)$ |
| $n \leq 10^6$ | $O(n \log n)$ |
| $n \leq 10^7$ | $O(n)$ 线性 |
| $n \leq 10^9$ | $O(\sqrt{n})$ / $O(\log n)$ |

### 5.3 比赛策略

1. **先想暴力：** 确保理解题意
2. **找规律：** 打表观察
3. **对拍验证：** 暴力 vs 优化，确保正确性
4. **注意取模：** 中间过程都取模
5. **多组数据清空：** 不要忘记重置全局变量
