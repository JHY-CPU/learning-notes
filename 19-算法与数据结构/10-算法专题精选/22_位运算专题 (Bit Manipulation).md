# 位运算专题 (Bit Manipulation)

## 一、概念定义与原理

### 1.1 基本位运算

| 运算 | 符号 | 说明 |
|------|------|------|
| 与 | `&` | 两位都为1时结果为1 |
| 或 | `|` | 有一位为1时结果为1 |
| 异或 | `^` | 两位不同时结果为1 |
| 取反 | `~` | 0变1，1变0 |
| 左移 | `<<` | 各位左移，低位补0 |
| 右移 | `>>` | 各位右移，高位补符号位 |

### 1.2 常用性质

- $a \oplus a = 0$（异或自身为0）
- $a \oplus 0 = a$（异或0不变）
- $a \& (a-1)$：消除 $a$ 的最低位 1
- $a \& (-a)$：取出 $a$ 的最低位 1（lowbit）
- $a \mid (a+1)$：将 $a$ 的最低位 0 变为 1

---

## 二、核心技巧

### 2.1 判断2的幂

$n > 0$ 且 $n \& (n-1) == 0$

### 2.2 统计1的个数

```cpp
int popcount(int x) { return __builtin_popcount(x); }
// 或手动：
int popcount(int x) {
    int c = 0;
    while (x) { x &= x-1; c++; }
    return c;
}
```

### 2.3 枚举子集

```cpp
for (int s = mask; s; s = (s-1) & mask) {
    // s 是 mask 的一个非空子集
}
```

### 2.4 状压DP

用二进制数表示集合状态，常见于 TSP、匹配等问题。

---

## 三、代码实现

### 3.1 基础操作 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 判断是否为2的幂
bool is_power_of_2(int n) { return n > 0 && (n & (n-1)) == 0; }

// 统计二进制中1的个数
int popcount(int n) { return __builtin_popcount(n); }

// 最低位1（lowbit）
int lowbit(int x) { return x & (-x); }

// 最高位1的位置（0-indexed）
int highest_bit(int x) { return 31 - __builtin_clz(x); }

// 交换两个数（不用临时变量）
void swap_xor(int& a, int& b) { a ^= b; b ^= a; a ^= b; }

// a+b 不用加法
int add(int a, int b) {
    while (b) { int carry = a & b; a ^= b; b = carry << 1; }
    return a;
}
```

### 3.2 枚举子集 - C++

```cpp
// 枚举 mask 的所有非空子集
void enumerate_subsets(int mask) {
    for (int s = mask; s; s = (s-1) & mask) {
        // s 是 mask 的子集
        cout << s << " ";
    }
}

// 枚举大小为 k 的子集
void enumerate_k_subsets(int n, int k) {
    int s = (1 << k) - 1;
    while (s < (1 << n)) {
        // s 是一个大小为 k 的子集
        cout << s << " ";
        int c = s & (-s);
        int r = s + c;
        s = (((r ^ s) >> 2) / c) | r;
    }
}
```

### 3.3 状压DP（旅行商问题）- C++

```cpp
// TSP：n 个城市，dist[i][j] 为距离，求最短哈密顿回路
int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX));
    dp[1][0] = 0; // 从城市0出发
    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u)) || dp[mask][u] == INT_MAX) continue;
            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v],
                                               dp[mask][u] + dist[u][v]);
            }
        }
    }
    int result = INT_MAX;
    for (int u = 0; u < n; u++)
        result = min(result, dp[(1<<n)-1][u] + dist[u][0]);
    return result;
}
```

### 3.4 Python 实现

```python
def is_power_of_2(n): return n > 0 and (n & (n-1)) == 0
def popcount(n): return bin(n).count('1')
def lowbit(x): return x & (-x)

# 枚举子集
def subsets(mask):
    s = mask
    while s:
        yield s
        s = (s - 1) & mask

# 状压DP - 集合覆盖
def set_cover(universe, sets):
    n = len(sets); target = (1 << len(universe)) - 1
    dp = [float('inf')] * (1 << len(universe))
    dp[0] = 0
    for mask in range(1 << len(universe)):
        if dp[mask] == float('inf'): continue
        for i, s in enumerate(sets):
            new_mask = mask | s
            dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
    return dp[target]

print(is_power_of_2(16))  # True
print(popcount(7))         # 3
print(list(subsets(0b1011)))  # [11, 10, 9, 8, 3, 2, 1, 0] 的子集
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 基本位运算 | $O(1)$ | |
| 枚举子集 | $O(2^k)$ | $k$ 为1的个数 |
| 状压DP | $O(2^n \cdot n)$ 或 $O(2^n \cdot n^2)$ | |

---

## 五、竞赛与面试应用场景

1. **LeetCode 136：** 只出现一次的数字（异或）
2. **LeetCode 191：** 位1的个数
3. **LeetCode 78：** 子集（位运算枚举）
4. **LeetCode 421：** 数组中两个数的最大异或值
5. **状压DP：** TSP、集合覆盖、棋盘问题
