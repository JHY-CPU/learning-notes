# 欧拉函数 (Euler's Totient Function)

## 一、概念定义与原理

### 1.1 定义

**欧拉函数** $\varphi(n)$ 定义为：小于等于 $n$ 的正整数中与 $n$ **互质**的数的个数。

$$\varphi(n) = |\{k \in [1, n] \mid \gcd(k, n) = 1\}|$$

**特殊值：**
- $\varphi(1) = 1$
- 若 $p$ 为质数，$\varphi(p) = p - 1$
- 若 $p$ 为质数且 $k \geq 1$，$\varphi(p^k) = p^k - p^{k-1} = p^{k-1}(p-1)$

### 1.2 积性函数性质

欧拉函数是**积性函数**：若 $\gcd(a, b) = 1$，则

$$\varphi(ab) = \varphi(a) \cdot \varphi(b)$$

### 1.3 一般公式

若 $n = p_1^{a_1} p_2^{a_2} \cdots p_k^{a_k}$，则：

$$\varphi(n) = n \prod_{i=1}^{k} \left(1 - \frac{1}{p_i}\right)$$

---

## 二、核心公式与性质

### 2.1 常用性质

1. **性质1：** $\sum_{d \mid n} \varphi(d) = n$（约数欧拉函数和等于 $n$）
2. **性质2：** 若 $p \nmid n$，则 $\varphi(pn) = \varphi(n) \cdot (p-1)$
3. **性质3：** 若 $p \mid n$，则 $\varphi(pn) = \varphi(n) \cdot p$
4. **性质4：** $n > 1$ 时，$\varphi(n)$ 为偶数

### 2.2 欧拉函数与欧拉定理的关系

欧拉定理：若 $\gcd(a, n) = 1$，则 $a^{\varphi(n)} \equiv 1 \pmod{n}$

---

## 三、代码实现

### 3.1 单个数的欧拉函数 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long euler_phi(long long n) {
    long long result = n;
    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) n /= i;
            result -= result / i;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}
```

### 3.2 线性筛求欧拉函数表 - C++

```cpp
const int MAXN = 1e7 + 5;
int phi_val[MAXN];
int primes[MAXN], cnt = 0;
bool is_composite[MAXN];

void euler_sieve_phi(int n) {
    phi_val[1] = 1;
    for (int i = 2; i <= n; i++) {
        if (!is_composite[i]) {
            primes[cnt++] = i;
            phi_val[i] = i - 1;
        }
        for (int j = 0; j < cnt && (long long)i * primes[j] <= n; j++) {
            is_composite[i * primes[j]] = true;
            if (i % primes[j] == 0) {
                phi_val[i * primes[j]] = phi_val[i] * primes[j];
                break;
            } else {
                phi_val[i * primes[j]] = phi_val[i] * (primes[j] - 1);
            }
        }
    }
}
```

### 3.3 Python 实现

```python
def euler_phi(n):
    result = n
    i = 2
    while i * i <= n:
        if n % i == 0:
            while n % i == 0:
                n //= i
            result -= result // i
        i += 1
    if n > 1:
        result -= result // n
    return result

def euler_sieve_phi(n):
    phi = [0] * (n + 1)
    primes = []
    is_composite = [False] * (n + 1)
    phi[1] = 1
    for i in range(2, n + 1):
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            if i * p > n: break
            is_composite[i * p] = True
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
    return phi
```

---

## 四、复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 单个计算 | $O(\sqrt{n})$ | $O(1)$ | 求单个 $\varphi(n)$ |
| 线性筛 | $O(N)$ | $O(N)$ | 求 $[1, N]$ 全部 $\varphi$ 值 |

---

## 五、竞赛与面试应用场景

### 5.1 互质对计数

$$\sum_{i=1}^{n} \sum_{j=1}^{n} [\gcd(i,j)=1] = 2\sum_{k=1}^{n}\varphi(k) - 1$$

### 5.2 降幂公式

$$a^b \equiv a^{b \bmod \varphi(m) + \varphi(m)} \pmod{m} \quad (b \geq \varphi(m))$$

### 5.3 求 [1, n] 中与 n 互质的数的和

结果 $= n \cdot \varphi(n) / 2$（$n > 1$）
