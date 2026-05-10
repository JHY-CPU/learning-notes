# 容斥原理 (Inclusion-Exclusion Principle)

## 一、概念定义与原理

### 1.1 基本思想

容斥原理用于计算多个集合的**并集**大小。核心思想是"奇加偶减"。

### 1.2 两个集合

$$|A \cup B| = |A| + |B| - |A \cap B|$$

### 1.3 三个集合

$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |B \cap C| - |A \cap C| + |A \cap B \cap C|$$

### 1.4 一般形式

$$\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \leq i_1 < \cdots < i_k \leq n} |A_{i_1} \cap \cdots \cap A_{i_k}|$$

---

## 二、核心应用

### 2.1 错排问题

$n$ 个元素的排列中，所有元素都不在原来位置上的排列数：

$$D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}$$

递推：$D_n = (n-1)(D_{n-1} + D_{n-2})$，$D_1 = 0, D_2 = 1$。

### 2.2 与某些数互质的数的个数

在 $[1, n]$ 中，与给定质数都互质的数的个数：

$$\text{count} = n - \sum \left\lfloor\frac{n}{p_i}\right\rfloor + \sum \left\lfloor\frac{n}{p_i p_j}\right\rfloor - \cdots$$

---

## 三、代码实现

### 3.1 容斥原理通用框架 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

long long inclusion_exclusion(long long n, vector<int>& primes) {
    int m = primes.size();
    long long result = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        long long lcm_val = 1;
        int bit_count = 0;
        for (int i = 0; i < m; i++) {
            if (mask & (1 << i)) {
                lcm_val *= primes[i];
                bit_count++;
                if (lcm_val > n) break;
            }
        }
        if (lcm_val > n) continue;
        if (bit_count & 1) result += n / lcm_val;
        else result -= n / lcm_val;
    }
    return result;
}
```

### 3.2 错排公式 - C++

```cpp
const int MAXN = 1e6 + 5;
long long derangement[MAXN];

void precompute_derangement(int n) {
    derangement[1] = 0;
    derangement[2] = 1;
    for (int i = 3; i <= n; i++) {
        derangement[i] = (i - 1) * (derangement[i-1] + derangement[i-2]) % MOD;
    }
}
```

### 3.3 Python 实现

```python
def inclusion_exclusion(n, primes):
    m = len(primes)
    result = 0
    for mask in range(1, 1 << m):
        lcm_val = 1
        bit_count = 0
        for i in range(m):
            if mask & (1 << i):
                lcm_val *= primes[i]
                bit_count += 1
                if lcm_val > n: break
        if lcm_val > n: continue
        if bit_count & 1: result += n // lcm_val
        else: result -= n // lcm_val
    return result

def count_coprime(n, primes):
    return n - inclusion_exclusion(n, primes)

def derangement(n):
    if n == 1: return 0
    if n == 2: return 1
    d = [0] * (n + 1)
    d[1], d[2] = 0, 1
    for i in range(3, n + 1):
        d[i] = (i - 1) * (d[i-1] + d[i-2])
    return d[n]

print(inclusion_exclusion(100, [2, 3, 5]))  # 74
print(count_coprime(100, [2, 3, 5]))        # 26
print(derangement(4))                        # 9
```

---

## 四、复杂度分析

| 应用 | 时间复杂度 | 说明 |
|------|-----------|------|
| 一般容斥 | $O(2^n)$ | 枚举所有子集 |
| 错排 | $O(n)$ | 线性递推 |

当质因子个数不超过 20 时，$O(2^k)$ 是可行的。

---

## 五、竞赛与面试应用场景

### 5.1 常见题型

1. **错排问题：** $n$ 个元素全不在原位的排列数
2. **互质计数：** 在 $[1, n]$ 中与若干数互质的个数
3. **不定方程：** 带上界约束的非负整数解计数
4. **多重约束：** 满足多个条件的方案数

### 5.2 注意事项

- 容斥时间复杂度 $O(2^n)$，适用于 $n \leq 20$
- 当需要对多查询使用容斥时，考虑莫比乌斯函数优化
- "奇加偶减"的符号不要搞错
