# 算法中的数学工具 (Math Tools)

## 一、常用求和公式

### 1.1 基本公式

```python
def sum_formulas(n):
    return {
        # 1 + 2 + ... + n
        "arithmetic": n * (n + 1) // 2,

        # 1² + 2² + ... + n²
        "squares": n * (n + 1) * (2*n + 1) // 6,

        # 1³ + 2³ + ... + n³
        "cubes": (n * (n + 1) // 2) ** 2,

        # 等比级数 1 + 2 + 4 + ... + 2^n
        "geometric_2": (1 << (n + 1)) - 1,  # 2^(n+1) - 1

        # 等比级数 a + ar + ar² + ... + ar^n
        "geometric": lambda a, r: a * (r**(n+1) - 1) // (r - 1) if r != 1 else a * (n+1),
    }
```

### 1.2 算法复杂度中的常见公式

| 复杂度 | 含义 | 公式 |
|--------|------|------|
| $O(n^2)$ | 两层循环 | $n(n-1)/2$ 次比较 |
| $O(n \log n)$ | 排序 | $n$ 层，每层 $n$ 次操作 |
| $O(2^n)$ | 子集 | $2^n$ 个子集 |
| $O(n!)$ | 排列 | $n!$ 种排列 |

---

## 二、模运算

### 2.1 基本性质

```python
MOD = 10**9 + 7

# (a + b) % MOD
def mod_add(a, b):
    return (a + b) % MOD

# (a - b) % MOD （处理负数）
def mod_sub(a, b):
    return (a - b + MOD) % MOD

# (a * b) % MOD
def mod_mul(a, b):
    return (a * b) % MOD

# (a / b) % MOD → 费马小定理: a/b = a * b^(MOD-2) mod MOD
def mod_div(a, b):
    return mod_mul(a, mod_pow(b, MOD - 2))
```

### 2.2 快速幂

```python
def mod_pow(base, exp, mod=10**9+7):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result
```

---

## 三、组合数学

### 3.1 排列与组合

```python
# 排列 P(n, k) = n! / (n-k)!
def permutation(n, k):
    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result

# 组合 C(n, k) = n! / (k! * (n-k)!)
def combination(n, k):
    if k > n or k < 0: return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

# 预处理阶乘（多次查询）
MOD = 10**9 + 7
MAXN = 2 * 10**5
fact = [1] * (MAXN + 1)
inv_fact = [1] * (MAXN + 1)

for i in range(1, MAXN + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact[MAXN] = pow(fact[MAXN], MOD - 2, MOD)
for i in range(MAXN - 1, -1, -1):
    inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

def comb(n, k):
    if k < 0 or k > n: return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
```

### 3.2 卡特兰数

$$C_n = \frac{1}{n+1}\binom{2n}{n} = \binom{2n}{n} - \binom{2n}{n+1}$$

应用：合法括号数、BST个数、三角剖分数。

```python
def catalan(n):
    return combination(2*n, n) // (n + 1)

# 前几项: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862...
```

---

## 四、GCD与扩展欧几里得

### 4.1 欧几里得算法

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)
```

### 4.2 扩展欧几里得

求 $ax + by = \gcd(a, b)$ 的整数解。

```python
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

# 求 a 模 m 的逆元
def mod_inverse(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1: return -1  # 无逆元
    return x % m
```

---

## 五、数论基础

### 5.1 素数筛

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(n+1) if is_prime[i]]
```

### 5.2 质因数分解

```python
def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

### 5.3 约数个数

$$\tau(n) = (e_1 + 1)(e_2 + 1)\cdots(e_k + 1)$$

其中 $n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k}$。

---

## 六、矩阵快速幂

用于加速线性递推。

```python
import numpy as np

def matrix_mult(A, B, mod=10**9+7):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
    return C

def matrix_pow(M, p, mod=10**9+7):
    n = len(M)
    result = [[int(i==j) for j in range(n)] for i in range(n)]
    while p > 0:
        if p & 1:
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)
        p >>= 1
    return result

# 求斐波那契第n项 O(log n)
# [F(n+1), F(n)] = [[1,1],[1,0]]^n * [1, 0]
def fibonacci(n):
    if n <= 1: return n
    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n - 1)
    return result[0][0]
```

---

## 七、复杂度总结

| 算法 | 时间 |
|------|------|
| GCD | $O(\log \min(a,b))$ |
| 快速幂 | $O(\log n)$ |
| 埃氏筛 | $O(n \log \log n)$ |
| 质因数分解 | $O(\sqrt{n})$ |
| 组合数(预处理) | $O(MAXN)$ 预处理 |
| 矩阵快速幂 | $O(n^3 \log p)$ |

---

## 八、面试要点

1. **模运算** — 大数取模、费马小定理
2. **快速幂** — 必须掌握
3. **组合数** — 预处理阶乘+逆元
4. **GCD** — 欧几里得算法
5. **矩阵快速幂** — 加速线性递推
