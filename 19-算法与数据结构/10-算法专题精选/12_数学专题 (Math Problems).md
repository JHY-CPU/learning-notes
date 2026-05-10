# 数学专题 (Math Problems)

## 一、概念定义与原理

本专题总结算法中常见的数学问题，涵盖数论、组合数学、概率期望等。

### 1.1 数论核心

- **质数：** 质数判定 $O(\sqrt{n})$、Miller-Rabin $O(k \log^2 n)$
- **GCD/LCM：** 欧几里得算法 $O(\log n)$
- **模运算：** 快速幂、费马求逆元、扩展欧几里得
- **同余方程：** CRT、EXCRT

### 1.2 组合数学核心

- **组合数：** 预处理阶乘、Lucas 定理
- **容斥原理：** 并集计数
- **卡特兰数：** $C_n = \frac{1}{n+1}\binom{2n}{n}$

### 1.3 期望与概率

- **期望线性性：** $E(X+Y) = E(X) + E(Y)$
- **期望DP：** 逆推期望

---

## 二、核心公式

### 2.1 快速幂

$$a^b \bmod m = \begin{cases} (a^{b/2})^2 \bmod m & b \text{为偶数} \\ a \cdot (a^{(b-1)/2})^2 \bmod m & b \text{为奇数} \end{cases}$$

### 2.2 模逆元

$$a^{-1} \equiv a^{p-2} \pmod{p} \quad (\text{费马小定理，} p \text{为质数})$$

### 2.3 组合数

$$C_n^m = \frac{n!}{m!(n-m)!} \equiv n! \cdot (m!)^{-1} \cdot ((n-m)!)^{-1} \pmod{p}$$

---

## 三、代码实现

### 3.1 数学工具箱 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;

long long power(long long a, long long b, long long m = MOD) {
    long long r = 1; a %= m;
    while (b) { if (b&1) r=r*a%m; a=a*a%m; b>>=1; }
    return r;
}

long long gcd(long long a, long long b) { return b ? gcd(b, a%b) : a; }
long long lcm(long long a, long long b) { return a / gcd(a,b) * b; }

long long mod_inv(long long a) { return power(a, MOD-2); }

// 组合数预处理
const int MAXN = 2e6 + 5;
long long fac[MAXN], inv_fac[MAXN];
void init_comb() {
    fac[0] = 1;
    for (int i = 1; i < MAXN; i++) fac[i] = fac[i-1]*i % MOD;
    inv_fac[MAXN-1] = power(fac[MAXN-1], MOD-2);
    for (int i = MAXN-2; i >= 0; i--) inv_fac[i] = inv_fac[i+1]*(i+1) % MOD;
}
long long C(int n, int m) {
    if (m < 0 || m > n) return 0;
    return fac[n]*inv_fac[m]%MOD*inv_fac[n-m]%MOD;
}
```

### 3.2 Python 实现

```python
MOD = 10**9 + 7

def power(a, b, m=MOD):
    r = 1; a %= m
    while b:
        if b & 1: r = r * a % m
        a = a * a % m; b >>= 1
    return r

from math import gcd

def lcm(a, b): return a // gcd(a, b) * b

# 组合数
from math import comb  # Python 3.8+

# 或手动实现
def C_mod(n, m, mod=MOD):
    if m < 0 or m > n: return 0
    result = 1
    for i in range(m):
        result = result * (n - i) % mod * power(i + 1, mod - 2) % mod
    return result

print(power(2, 100))      # 2^100 mod 10^9+7
print(C_mod(100, 50))     # C(100,50) mod 10^9+7
print(gcd(12, 18))        # 6
print(lcm(12, 18))        # 36
```

### 3.3 期望DP模板

```cpp
// 掷骰子直到和 >= n 的期望次数
double expected_rolls(int n) {
    vector<double> E(n + 6, 0);
    for (int i = n - 1; i >= 0; i--) {
        E[i] = 1.0;
        for (int j = 1; j <= 6; j++) E[i] += E[i+j] / 6.0;
    }
    return E[0];
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 |
|------|-----------|
| 快速幂 | $O(\log b)$ |
| GCD | $O(\log(\min(a,b)))$ |
| 组合数（预处理） | $O(N)$ 预处理，$O(1)$ 查询 |
| 组合数（单次） | $O(m)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 50：** Pow(x, n)（快速幂）
2. **LeetCode 62：** 不同路径（组合数）
3. **大数取模：** 费马小定理
4. **期望计算：** 期望DP
5. **卡特兰数：** 括号匹配、出栈序列
