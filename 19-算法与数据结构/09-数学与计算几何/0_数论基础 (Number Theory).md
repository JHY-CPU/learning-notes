# 数论基础 (Number Theory)

## 一、概念定义与原理

数论是研究整数性质的数学分支，在算法竞赛和密码学中有广泛应用。本节介绍整除、质数、质因数分解、最大公约数（GCD）和最小公倍数（LCM）等核心概念。

### 1.1 整除

若整数 $a$ 除以整数 $b$（$b \neq 0$）所得商为整数且余数为零，则称 $b$ **整除** $a$，记作 $b \mid a$。

**基本性质：**
- 若 $a \mid b$ 且 $b \mid c$，则 $a \mid c$（传递性）
- 若 $a \mid b$ 且 $a \mid c$，则 $a \mid (b \pm c)$
- 若 $a \mid b$ 且 $a \mid c$，则 $a \mid (mb + nc)$，$\forall m, n \in \mathbb{Z}$

### 1.2 质数与合数

**定义：** 大于1的自然数中，除了1和它本身以外不再有其他因数的数称为**质数**（素数）。大于1的非质数自然数称为**合数**。

**基本定理（算术基本定理）：** 任意大于1的正整数都能唯一地分解为质数的乘积：

$$n = p_1^{a_1} \cdot p_2^{a_2} \cdots p_k^{a_k}$$

其中 $p_1 < p_2 < \cdots < p_k$ 为质数，$a_i \geq 1$。

### 1.3 带余除法

对于整数 $a$ 和正整数 $b$，存在唯一的整数 $q$（商）和 $r$（余数），使得：

$$a = qb + r, \quad 0 \leq r < b$$

---

## 二、核心算法与公式

### 2.1 最大公约数 (GCD)

**欧几里得算法（辗转相除法）：**

$$\gcd(a, b) = \gcd(b, a \bmod b)$$

终止条件：$\gcd(a, 0) = a$

**性质：**
- $\gcd(a, b) = \gcd(b, a)$
- $\gcd(a, b) = \gcd(a, b - a)$（更相减损术）
- $\gcd(ka, kb) = k \cdot \gcd(a, b)$
- $\gcd(a, b) \cdot \text{lcm}(a, b) = a \cdot b$（仅对正整数成立）

### 2.2 最小公倍数 (LCM)

$$\text{lcm}(a, b) = \frac{a \cdot b}{\gcd(a, b)}$$

**注意：** 当 $a, b$ 较大时，先除后乘防止溢出：$\text{lcm}(a, b) = a / \gcd(a, b) \cdot b$

### 2.3 快速幂

计算 $a^b \bmod m$，将 $b$ 按二进制拆分：

```cpp
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
```

### 2.4 质因数分解

**试除法：** 从2到 $\sqrt{n}$ 逐个试除，时间复杂度 $O(\sqrt{n})$。

---

## 三、代码实现

### 3.1 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 欧几里得算法求GCD
long long gcd(long long a, long long b) {
    return b == 0 ? a : gcd(b, a % b);
}

// 利用GCD求LCM
long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}

// 质因数分解
map<long long, int> factorize(long long n) {
    map<long long, int> factors;
    for (long long i = 2; i * i <= n; i++) {
        while (n % i == 0) {
            factors[i]++;
            n /= i;
        }
    }
    if (n > 1) factors[n]++;
    return factors;
}

// 质数判定
bool isPrime(long long n) {
    if (n < 2) return false;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}
```

### 3.2 Python 实现

```python
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a // math.gcd(a, b) * b

def factorize(n):
    factors = {}
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def power(a, b, m):
    r = 1; a %= m
    while b:
        if b & 1: r = r * a % m
        a = a * a % m; b >>= 1
    return r
```

### 3.3 约数枚举

```cpp
vector<long long> getDivisors(long long n) {
    vector<long long> divisors;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) divisors.push_back(n / i);
        }
    }
    sort(divisors.begin(), divisors.end());
    return divisors;
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 欧几里得GCD | $O(\log(\min(a,b)))$ | $O(1)$ | 非常高效 |
| LCM | $O(\log(\min(a,b)))$ | $O(1)$ | 依赖GCD |
| 快速幂 | $O(\log b)$ | $O(1)$ | 二进制拆分 |
| 试除法分解 | $O(\sqrt{n})$ | $O(\log n)$ | 因子数 |
| 质数判定 | $O(\sqrt{n})$ | $O(1)$ | 试除法 |

---

## 五、竞赛与面试应用场景

### 5.1 约数个数与约数和公式

若 $n = p_1^{a_1} \cdot p_2^{a_2} \cdots p_k^{a_k}$，则：
- 约数个数：$\tau(n) = (a_1 + 1)(a_2 + 1) \cdots (a_k + 1)$
- 约数之和：$\sigma(n) = \prod_{i=1}^{k} \frac{p_i^{a_i+1} - 1}{p_i - 1}$

### 5.2 常见题型

1. **求GCD/LCM：** 给定多个数求最大公约数或最小公倍数
2. **质因数分解：** 求约数个数、约数之和
3. **互质判断：** $\gcd(a, b) = 1$ 时 $a, b$ 互质
4. **快速幂取模：** $a^b \bmod m$ 的高效计算

### 5.3 裴蜀定理

对于任意正整数 $a, b$，存在整数 $x, y$ 使得 $ax + by = \gcd(a, b)$。更一般地，$ax + by = c$ 有整数解当且仅当 $\gcd(a, b) \mid c$。
