# 0-1背包 (0-1 Knapsack)

## 一、问题定义

### 1.1 经典描述

有 $n$ 件物品和一个容量为 $W$ 的背包。第 $i$ 件物品的重量为 $w_i$，价值为 $v_i$。每件物品**最多选一次**，求背包能装的最大价值。

### 1.2 数学形式

$$\max \sum_{i=1}^{n} v_i \cdot x_i \quad \text{s.t.} \quad \sum_{i=1}^{n} w_i \cdot x_i \leq W, \quad x_i \in \{0, 1\}$$

---

## 二、DP解法

### 2.1 二维DP

**状态定义：** `dp[i][j]` = 前 `i` 件物品，容量 `j` 的最大价值。

**状态转移：**

$$dp[i][j] = \max(dp[i-1][j], \quad dp[i-1][j-w_i] + v_i)$$

（不选第i件 vs 选第i件）

```python
def knapsack_01_2d(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, W + 1):
            dp[i][j] = dp[i-1][j]  # 不选
            if j >= weights[i-1]:
                dp[i][j] = max(dp[i][j],
                    dp[i-1][j - weights[i-1]] + values[i-1])  # 选

    return dp[n][W]
```

**时间复杂度：** $O(nW)$
**空间复杂度：** $O(nW)$

### 2.2 一维DP（空间优化）

关键：**逆序遍历**，保证每件物品只被选一次。

```python
def knapsack_01(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for j in range(W, weights[i] - 1, -1):  # 逆序！
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[W]
```

**为什么逆序？** 逆序保证 `dp[j-w_i]` 还是上一轮（未选第i件物品）的值。如果正序，`dp[j-w_i]` 可能已被当前轮次更新过，等同于允许重复选。

**时间复杂度：** $O(nW)$
**空间复杂度：** $O(W)$

---

## 三、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 二维DP
int knapsack2D(vector<int>& w, vector<int>& v, int W) {
    int n = w.size();
    vector<vector<int>> dp(n+1, vector<int>(W+1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 0; j <= W; j++) {
            dp[i][j] = dp[i-1][j];
            if (j >= w[i-1])
                dp[i][j] = max(dp[i][j], dp[i-1][j-w[i-1]] + v[i-1]);
        }
    return dp[n][W];
}

// 一维优化
int knapsack(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++)
        for (int j = W; j >= w[i]; j--)  // 逆序
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    return dp[W];
}
```

---

## 四、方案还原

记录每个状态的选择，从 `dp[n][W]` 倒推。

```python
def knapsack_with_items(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(W + 1):
            dp[i][j] = dp[i-1][j]
            if j >= weights[i-1]:
                dp[i][j] = max(dp[i][j],
                    dp[i-1][j-weights[i-1]] + values[i-1])

    # 还原选择的物品
    items = []
    j = W
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i-1][j]:
            items.append(i - 1)
            j -= weights[i-1]

    return dp[n][W], items[::-1]
```

---

## 五、完全背包 vs 0-1 背包

| 区别 | 0-1 背包 | 完全背包 |
|------|---------|---------|
| 每个物品 | 最多选一次 | 可选无限次 |
| 内层循环 | **逆序** W -> w[i] | **正序** w[i] -> W |
| 转移方程 | 相同 | 相同，但正序实现 |

---

## 六、变种问题

### 6.1 刚好装满 (LeetCode 416)

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

### 6.2 方案计数 (LeetCode 494)

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if (total + target) % 2 or total < abs(target):
        return 0
    subset_sum = (total + target) // 2
    dp = [0] * (subset_sum + 1)
    dp[0] = 1
    for num in nums:
        for j in range(subset_sum, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[subset_sum]
```

### 6.3 多重背包

每个物品有数量限制，用二进制拆分转为01背包：

```python
def knapsack_bounded(weights, values, counts, W):
    # 二进制拆分
    new_w, new_v = [], []
    for w, v, c in zip(weights, values, counts):
        k = 1
        while k < c:
            new_w.append(w * k)
            new_v.append(v * k)
            c -= k
            k *= 2
        if c > 0:
            new_w.append(w * c)
            new_v.append(v * c)

    # 01背包
    dp = [0] * (W + 1)
    for w, v in zip(new_w, new_v):
        for j in range(W, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

### 6.4 分组背包

每组选一个物品或不选：

```python
def knapsack_grouped(groups, W):
    """groups: [(weight, value), ...] per group"""
    dp = [0] * (W + 1)
    for group in groups:
        for j in range(W, -1, -1):
            for w, v in group:
                if j >= w:
                    dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

---

## 七、复杂度分析

| 变种 | 时间 | 空间 |
|------|------|------|
| 基本01背包 | $O(nW)$ | $O(W)$ |
| 方案还原 | $O(nW)$ | $O(nW)$ |
| 多重背包(二进制) | $O(nW \log C)$ | $O(W)$ |
| 分组背包 | $O(nW)$ | $O(W)$ |
| 刚好装满 | $O(nW)$ | $O(W)$ |

---

## 八、面试要点

1. **逆序遍历** — 01背包的灵魂，保证每个物品只选一次
2. **初始化** — 求max初始化0，求min初始化inf
3. **转化为背包** — 很多问题本质是背包（子集和、目标和等）
4. **方案还原** — 需要二维DP才能还原
5. **正序 vs 逆序** — 区分01背包和完全背包的关键
