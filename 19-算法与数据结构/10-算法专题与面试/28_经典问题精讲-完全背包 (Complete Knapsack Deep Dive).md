# 经典问题精讲-完全背包 (Complete Knapsack Deep Dive)

## 一、背包问题分类

| 类型 | 每件物品数量 | 状态转移 |
|------|------------|---------|
| 01背包 | 最多1个 | `dp[j] = max(dp[j], dp[j-w]+v)` 逆序 |
| 完全背包 | 无限个 | `dp[j] = max(dp[j], dp[j-w]+v)` 正序 |
| 多重背包 | 有限个 | 二进制拆分 |
| 分组背包 | 每组选1个 | 组内01背包 |

---

## 二、完全背包详解

### 2.1 二维DP

`dp[i][j]` = 前 `i` 件物品，容量 `j` 的最大价值。

```python
def knapsack_complete_2d(weights, values, W):
    n = len(weights)
    dp = [[0]*(W+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, W+1):
            dp[i][j] = dp[i-1][j]  # 不选
            if j >= weights[i-1]:
                dp[i][j] = max(dp[i][j],
                    dp[i][j - weights[i-1]] + values[i-1])  # 可重复选

    return dp[n][W]
```

### 2.2 一维DP（空间优化）

关键区别：完全背包**正序**遍历（允许重复选），01背包**逆序**遍历。

```python
def knapsack_complete(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for j in range(w, W + 1):  # 正序！
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

**为什么正序？** 正序时 `dp[j-w]` 可能已经包含了当前物品（本次迭代更新过的），即允许重复选。

---

## 三、经典变种

### 3.1 零钱兑换I (LeetCode 322) — 最少硬币数

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for j in range(c, amount + 1):
            dp[j] = min(dp[j], dp[j - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 3.2 零钱兑换II (LeetCode 518) — 组合数

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:           # 先遍历物品 → 组合数（不计顺序）
        for j in range(c, amount + 1):
            dp[j] += dp[j - c]
    return dp[amount]
```

**区分组合和排列：**
- 组合（不计顺序）：先遍历物品，后遍历容量
- 排列（计顺序）：先遍历容量，后遍历物品

### 3.3 排列数 — 先容量后物品

```python
# 用硬币凑金额，不同顺序算不同方案
def change_permutation(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for j in range(1, amount + 1):    # 先遍历容量
        for c in coins:                 # 后遍历物品
            if j >= c:
                dp[j] += dp[j - c]
    return dp[amount]
```

### 3.4 完全平方数 (LeetCode 279)

```python
def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, int(n**0.5) + 1):
        sq = i * i
        for j in range(sq, n + 1):
            dp[j] = min(dp[j], dp[j - sq] + 1)
    return dp[n]
```

### 3.5 单词拆分 (LeetCode 139)

```python
def word_break(s, word_dict):
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]
```

---

## 四、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 完全背包
int knapsackComplete(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++)
        for (int j = w[i]; j <= W; j++)  // 正序
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    return dp[W];
}

// 零钱兑换 — 最少硬币
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int c : coins)
        for (int j = c; j <= amount; j++)
            dp[j] = min(dp[j], dp[j - c] + 1);
    return dp[amount] > amount ? -1 : dp[amount];
}

// 零钱兑换II — 组合数
int change(int amount, vector<int>& coins) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int c : coins)
        for (int j = c; j <= amount; j++)
            dp[j] += dp[j - c];
    return dp[amount];
}
```

---

## 五、复杂度分析

| 问题 | 时间 | 空间 |
|------|------|------|
| 完全背包 | $O(nW)$ | $O(W)$ |
| 零钱兑换I | $O(n \cdot amount)$ | $O(amount)$ |
| 零钱兑换II | $O(n \cdot amount)$ | $O(amount)$ |
| 完全平方数 | $O(n\sqrt{n})$ | $O(n)$ |
| 单词拆分 | $O(n^2)$ | $O(n)$ |

---

## 六、面试要点

1. **正序 vs 逆序** — 完全背包正序，01背包逆序
2. **组合 vs 排列** — 遍历顺序不同
3. **初始化** — 最小值问题初始化 inf，计数问题初始化 0/1
4. **目标判断** — 是否等于 / 至少 / 至多 / 恰好
