# 筛法 (Sieve Methods)

## 一、概念定义与原理

筛法是批量求素数的经典算法。其核心思想是：从2开始，依次标记每个素数的倍数为合数，最终未被标记的即为素数。

### 1.1 问题背景

给定上界 $N$，求 $[2, N]$ 中所有素数：
- **试除法逐个判定：** $O(N\sqrt{N})$
- **埃氏筛：** $O(N \log \log N)$
- **欧拉筛（线性筛）：** $O(N)$

### 1.2 基本思想

利用"合数一定有质因子"这一事实，用已知的质数去筛除其倍数，从而高效地找出所有素数。

---

## 二、核心算法

### 2.1 埃氏筛（Sieve of Eratosthenes）

**算法步骤：**
1. 初始化布尔数组 `is_prime[0..N]`，全部标记为 `true`
2. 从 $i = 2$ 开始遍历到 $\sqrt{N}$
3. 若 `is_prime[i] == true`，则将 $i^2, i^2+i, i^2+2i, \ldots \leq N$ 全部标记为 `false`
4. 遍历结束后，`is_prime[i] == true` 的位置即为素数

**关键优化：** 从 $i^2$ 开始筛而非从 $2i$ 开始，因为更小的倍数已被筛过。

### 2.2 欧拉筛（线性筛）

埃氏筛会重复标记合数。欧拉筛保证每个合数只被其**最小质因子**筛掉一次。

**核心原则：** 对于当前遍历到的数 $i$ 和已知素数 $p_j$：
- 将 $i \cdot p_j$ 标记为合数
- 若 $p_j \mid i$，则停止

---

## 三、代码实现

### 3.1 埃氏筛 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1e7 + 5;
bool is_prime[MAXN];
vector<int> primes;

void eratosthenes(int n) {
    fill(is_prime, is_prime + n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) primes.push_back(i);
    }
}
```

### 3.2 欧拉筛 - C++

```cpp
const int MAXN = 1e7 + 5;
bool is_prime[MAXN];
int min_factor[MAXN];
vector<int> primes;

void euler_sieve(int n) {
    fill(is_prime, is_prime + n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            primes.push_back(i);
            min_factor[i] = i;
        }
        for (int j = 0; j < primes.size() && (long long)i * primes[j] <= n; j++) {
            is_prime[i * primes[j]] = false;
            min_factor[i * primes[j]] = primes[j];
            if (i % primes[j] == 0) break;
        }
    }
}
```

### 3.3 Python 实现

```python
def eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def euler_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes
```

### 3.4 区间筛（Segmented Sieve）

当 $[L, R]$ 很大但 $R - L$ 不太大时，先筛 $[1, \sqrt{R}]$ 的素数，再筛 $[L, R]$。

```cpp
vector<long long> segmented_sieve(long long L, long long R) {
    int limit = (int)sqrt(R) + 1;
    vector<int> small_primes = euler_sieve(limit);
    vector<bool> is_prime(R - L + 1, true);
    if (L == 1) is_prime[0] = false;
    for (int p : small_primes) {
        long long start = max((long long)p * p, ((L + p - 1) / p) * p);
        for (long long j = start; j <= R; j += p) {
            is_prime[j - L] = false;
        }
    }
    vector<long long> result;
    for (long long i = 0; i <= R - L; i++) {
        if (is_prime[i]) result.push_back(L + i);
    }
    return result;
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 特点 |
|------|-----------|-----------|------|
| 埃氏筛 | $O(N \log \log N)$ | $O(N)$ | 实现简单，常数小 |
| 欧拉筛 | $O(N)$ | $O(N)$ | 线性，可同时求最小质因子 |
| 区间筛 | $O((R-L+1) \cdot \log\log\sqrt{R})$ | $O(\sqrt{R} + (R-L))$ | 适合大区间 |

---

## 五、竞赛与面试应用场景

### 5.1 利用欧拉筛做快速质因数分解

```cpp
vector<pair<int,int>> fast_factorize(int x) {
    vector<pair<int,int>> res;
    while (x > 1) {
        int p = min_factor[x];
        int cnt = 0;
        while (x % p == 0) { x /= p; cnt++; }
        res.push_back({p, cnt});
    }
    return res;
}
// 时间复杂度 O(log n)
```

### 5.2 注意事项

- 埃氏筛从 $i^2$ 开始筛，不是从 $2i$
- 欧拉筛中 `i % primes[j] == 0` 的判断是线性复杂度的关键
- 区间筛中 $L$ 可能为 $0$ 或 $1$，需要特殊处理
- 注意 `long long` 溢出：`i * primes[j]` 需要强转
