# 数论函数 (Number Theoretic Functions)

## 一、概念定义与原理

### 1.1 积性函数

**定义：** 若 $\gcd(a, b) = 1$ 时 $f(ab) = f(a) \cdot f(b)$，则 $f$ 为**积性函数**。

常见积性函数：
- 欧拉函数 $\varphi(n)$
- 莫比乌斯函数 $\mu(n)$
- 约数个数 $d(n) = \tau(n)$
- 约数之和 $\sigma(n)$

### 1.2 莫比乌斯函数

$$\mu(n) = \begin{cases} 1 & n = 1 \\ (-1)^k & n = p_1 p_2 \cdots p_k \text{（k个不同质数）} \\ 0 & n \text{有平方因子} \end{cases}$$

**关键性质：** $\sum_{d \mid n} \mu(d) = [n = 1]$

### 1.3 狄利克雷卷积

$$(f * g)(n) = \sum_{d \mid n} f(d) \cdot g(n/d)$$

**莫比乌斯反演：** 若 $g(n) = \sum_{d \mid n} f(d)$，则 $f(n) = \sum_{d \mid n} \mu(d) \cdot g(n/d)$

---

## 二、核心公式

### 2.1 莫比乌斯反演

$$g(n) = \sum_{d|n} f(d) \Leftrightarrow f(n) = \sum_{d|n} \mu(d) \cdot g\left(\frac{n}{d}\right)$$

等价形式：$f(n) = \sum_{d|n} \mu\left(\frac{n}{d}\right) \cdot g(d)$

### 2.2 经典应用

$\sum_{i=1}^{n} \sum_{j=1}^{n} [\gcd(i,j)=1] = \sum_{d=1}^{n} \mu(d) \cdot \lfloor n/d \rfloor^2$

---

## 三、代码实现

### 3.1 线性筛求莫比乌斯函数 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1e7 + 5;
int mu[MAXN], primes[MAXN], cnt = 0;
bool is_composite[MAXN];

void sieve_mu(int n) {
    mu[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!is_composite[i]) {
            primes[cnt++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < cnt && (long long)i * primes[j] <= n; j++) {
            is_composite[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                mu[i * primes[j]] = 0; // 有平方因子
                break;
            } else {
                mu[i * primes[j]] = -mu[i];
            }
        }
    }
}
```

### 3.2 莫比乌斯反演应用

```cpp
// 求 [1,n] 中 gcd(i,j)=k 的对数
// 即 sum mu(d) * floor(n/d)^2
long long coprime_pairs(long long n) {
    long long result = 0;
    for (long long d = 1; d <= n; d++) {
        result += mu[d] * (n / d) * (n / d);
    }
    return result;
}

// 分块优化：O(sqrt(n))
long long coprime_pairs_fast(long long n) {
    long long result = 0;
    for (long long l = 1, r; l <= n; l = r + 1) {
        r = n / (n / l);
        // 需要预处理 mu 的前缀和
        result += (mu_prefix[r] - mu_prefix[l-1]) * (n / l) * (n / l);
    }
    return result;
}
```

### 3.3 Python 实现

```python
def sieve_mu(n):
    mu = [0] * (n + 1)
    primes = []
    is_comp = [False] * (n + 1)
    mu[1] = 1
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def coprime_pairs(n):
    mu = sieve_mu(n)
    return sum(mu[d] * (n // d) ** 2 for d in range(1, n + 1))

print(coprime_pairs(10))  # gcd=1的对数
```

### 3.4 约数函数

```cpp
// 约数个数 d(n) = tau(n)
// 如果 n = p1^a1 * p2^a2 * ...，则 d(n) = (a1+1)(a2+1)...
int divisor_count(int n) {
    int result = 1;
    for (int i = 2; i * i <= n; i++) {
        int cnt = 0;
        while (n % i == 0) { n /= i; cnt++; }
        result *= (cnt + 1);
    }
    if (n > 1) result *= 2;
    return result;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 线性筛 $\mu$ | $O(N)$ | 预处理 |
| 莫比乌斯反演暴力 | $O(n)$ | 单次查询 |
| 分块优化 | $O(\sqrt{n})$ | 需前缀和 |
| 约数个数 | $O(\sqrt{n})$ | 分解质因数 |

---

## 五、竞赛与面试应用场景

1. **互质对计数：** 莫比乌斯反演
2. **容斥优化：** 用 $\mu$ 函数代替 $O(2^k)$ 容斥
3. **积性函数前缀和：** 杜教筛
4. **狄利克雷卷积：** 函数变换
