# 欧拉定理与费马小定理 (Euler's Theorem & Fermat's Little Theorem)

## 一、概念定义与原理

### 1.1 费马小定理 (Fermat's Little Theorem)

**定理：** 若 $p$ 为质数，且 $\gcd(a, p) = 1$，则：

$$a^{p-1} \equiv 1 \pmod{p}$$

等价形式：$a^p \equiv a \pmod{p}$（此形式不需要 $\gcd(a, p) = 1$）。

### 1.2 欧拉定理 (Euler's Theorem)

**定理：** 若 $\gcd(a, n) = 1$，则：

$$a^{\varphi(n)} \equiv 1 \pmod{n}$$

费马小定理是欧拉定理的特例：当 $n = p$（质数）时，$\varphi(p) = p - 1$。

### 1.3 降幂公式（扩展欧拉定理）

对于任意 $a, m$ 和足够大的 $b$：

$$a^b \equiv a^{b \bmod \varphi(m) + \varphi(m)} \pmod{m} \quad (b \geq \varphi(m))$$

注意不要求 $\gcd(a, m) = 1$。

---

## 二、核心应用

### 2.1 快速幂取模

利用费马小定理将大指数化简：

$$a^b \bmod p = a^{b \bmod (p-1)} \bmod p \quad (p \text{为质数}, \gcd(a, p)=1)$$

### 2.2 模逆元

**费马小定理求逆元：** 当 $p$ 为质数时：

$$a^{-1} \equiv a^{p-2} \pmod{p}$$

**扩展欧几里得求逆元：** 由 $ax + my = 1$ 得 $ax \equiv 1 \pmod{m}$，$x$ 即为逆元。

---

## 三、代码实现

### 3.1 快速幂与费马求逆元 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long power(long long a, long long b, long long m) {
    long long result = 1;
    a %= m;
    while (b > 0) {
        if (b & 1) result = result * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return result;
}

// 费马小定理求逆元 (p必须是质数)
long long fermat_inv(long long a, long long p) {
    return power(a, p - 2, p);
}
```

### 3.2 线性求逆元表

```cpp
const int MAXN = 1e6 + 5;
const int MOD = 1e9 + 7;
long long inv[MAXN];

void precompute_inv(int n) {
    inv[1] = 1;
    for (int i = 2; i <= n; i++) {
        inv[i] = (MOD - MOD / i) * inv[MOD % i] % MOD;
    }
}
// 公式：k*i + r ≡ 0 (mod MOD) → 1/i ≡ -(MOD/i) * inv[MOD%i] (mod MOD)
```

### 3.3 Python 实现

```python
def power(a, b, m):
    result = 1; a %= m
    while b > 0:
        if b & 1: result = result * a % m
        a = a * a % m; b >>= 1
    return result

def fermat_inv(a, p):
    return power(a, p - 2, p)

MOD = 10**9 + 7
print(power(2, 100, MOD))    # 2^100 mod (10^9+7)
print(fermat_inv(3, MOD))    # 3 在 MOD 下的逆元
```

### 3.4 组合数取模模板

```cpp
const int MOD = 1e9 + 7;
const int MAXN = 2e6 + 5;
long long fac[MAXN], inv_fac[MAXN];

void precompute() {
    fac[0] = 1;
    for (int i = 1; i < MAXN; i++) fac[i] = fac[i-1] * i % MOD;
    inv_fac[MAXN-1] = power(fac[MAXN-1], MOD-2, MOD);
    for (int i = MAXN-2; i >= 0; i--) inv_fac[i] = inv_fac[i+1] * (i+1) % MOD;
}

long long C(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fac[n] % MOD * inv_fac[k] % MOD * inv_fac[n-k] % MOD;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 快速幂 | $O(\log b)$ | 二进制分解指数 |
| 费马求逆元 | $O(\log p)$ | 一次快速幂 |
| 线性求逆元表 | $O(n)$ | 预处理后 $O(1)$ 查询 |
| 降幂计算 | $O(\log m + \log b)$ | 需先计算 $\varphi(m)$ |

---

## 五、竞赛与面试应用场景

### 5.1 典型应用

1. **组合数取模：** $C_n^k = \frac{n!}{k!(n-k)!} \bmod p$，需要阶乘逆元
2. **大指数取模：** $a^{10^{18}} \bmod p$，利用费马小定理降幂
3. **超级幂取模：** $2^{3^{4^{...}}} \bmod m$，利用降幂公式递归
4. **除法取模：** $a/b \bmod p = a \cdot b^{-1} \bmod p$

### 5.2 注意事项

- 费马小定理要求模数为**质数**
- 降幂公式中指数需要加 $\varphi(m)$ 的条件是 $b \geq \varphi(m)$
- 线性求逆元只适用于模数为质数的情况
