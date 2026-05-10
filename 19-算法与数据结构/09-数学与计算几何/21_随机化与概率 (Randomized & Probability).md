# 随机化与概率 (Randomized & Probability)

## 一、概念定义与原理

### 1.1 随机化算法分类

- **Las Vegas算法：** 结果总是正确的，但运行时间是随机的（如随机快速排序）
- **Monte Carlo算法：** 运行时间是确定的，但可能以一定概率给出错误结果

### 1.2 期望复杂度

算法的期望运行时间是在所有随机选择上的平均运行时间。

### 1.3 随机化的常见用途

- 避免最坏情况（如快速排序的随机化选主元）
- 哈希函数（减少冲突）
- 随机采样和概率验证

---

## 二、核心算法

### 2.1 随机快速排序

随机选择主元，期望时间复杂度 $O(n \log n)$，最坏情况的概率极低。

### 2.2 随机增量法（最小圆覆盖）

Welzl 算法：随机增量法以期望 $O(n)$ 时间求最小覆盖圆。

### 2.3 Miller-Rabin 质数测试

概率性质数判定，单次测试错误率 $\leq 1/4$，$k$ 次测试错误率 $\leq 4^{-k}$。

### 2.4 Pollard's Rho 分解

利用随机化和 Floyd 环检测在期望 $O(n^{1/4})$ 时间内找到非平凡因子。

---

## 三、代码实现

### 3.1 随机快速排序 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

void randomized_quicksort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot_idx = uniform_int_distribution<int>(l, r)(rng);
    swap(a[l], a[pivot_idx]);
    int pivot = a[l];
    int i = l, j = r;
    while (i < j) {
        while (i < j && a[j] >= pivot) j--;
        a[i] = a[j];
        while (i < j && a[i] <= pivot) i++;
        a[j] = a[i];
    }
    a[i] = pivot;
    randomized_quicksort(a, l, i - 1);
    randomized_quicksort(a, i + 1, r);
}
```

### 3.2 Miller-Rabin 质数测试

```cpp
long long power(long long a, long long b, long long m) {
    long long r = 1; a %= m;
    while (b) { if (b & 1) r = (__int128)r * a % m; a = (__int128)a * a % m; b >>= 1; }
    return r;
}

bool miller_rabin(long long n, long long a) {
    if (n % a == 0) return n == a;
    long long d = n - 1;
    while (d % 2 == 0) d /= 2;
    long long x = power(a, d, n);
    if (x == 1 || x == n - 1) return true;
    while (d < n - 1) {
        x = (__int128)x * x % n;
        d *= 2;
        if (x == n - 1) return true;
    }
    return false;
}

bool is_prime(long long n) {
    if (n < 2) return false;
    for (long long a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (n == a) return true;
        if (!miller_rabin(n, a)) return false;
    }
    return true;
}
```

### 3.3 Python 实现

```python
import random

def miller_rabin(n, a):
    if n % a == 0: return n == a
    d = n - 1
    while d % 2 == 0: d //= 2
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return True
    while d < n - 1:
        x = x * x % n
        d *= 2
        if x == n - 1: return True
    return False

def is_prime(n):
    if n < 2: return False
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n == a: return True
        if not miller_rabin(n, a): return False
    return True

def randomized_select(a, k):
    """随机化选择第k小元素 O(n) 期望"""
    if len(a) == 1: return a[0]
    pivot = random.choice(a)
    less = [x for x in a if x < pivot]
    equal = [x for x in a if x == pivot]
    greater = [x for x in a if x > pivot]
    if k < len(less): return randomized_select(less, k)
    elif k < len(less) + len(equal): return pivot
    else: return randomized_select(greater, k - len(less) - len(equal))

print(is_prime(10**18 + 7))  # 可以快速判定大数
```

### 3.4 随机哈希防卡

```cpp
// 随机化哈希基数防 hack
struct StringHash {
    static const long long BASE = 19260817 + rng() % 100000;
    static const long long MOD = 1e9 + 9;
    vector<long long> h, pw;
    void init(const string& s) {
        int n = s.size();
        h.resize(n + 1); pw.resize(n + 1);
        pw[0] = 1;
        for (int i = 0; i < n; i++) {
            h[i+1] = (h[i] * BASE + s[i]) % MOD;
            pw[i+1] = pw[i] * BASE % MOD;
        }
    }
    long long get(int l, int r) {
        return (h[r+1] - h[l] * pw[r-l+1] % MOD + MOD) % MOD;
    }
};
```

---

## 四、复杂度分析

| 算法 | 期望时间复杂度 | 最坏时间复杂度 |
|------|-------------|-------------|
| 随机快速排序 | $O(n \log n)$ | $O(n^2)$（概率极低） |
| Miller-Rabin | $O(k \log^2 n)$ | 同期望 |
| Pollard's Rho | $O(n^{1/4})$ | $O(\sqrt{n})$ |
| 随机化选择 | $O(n)$ | $O(n^2)$ |

---

## 五、竞赛与面试应用场景

1. **防卡哈希：** 随机化哈希基数
2. **质数判定：** Miller-Rabin 用于大数
3. **随机化技巧：** 随机排序防特殊数据
4. **概率验证：** 字符串匹配等
5. **随机增量法：** 几何中的最小覆盖问题
