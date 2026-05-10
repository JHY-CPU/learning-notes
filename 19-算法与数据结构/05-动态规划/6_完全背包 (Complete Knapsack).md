# 完全背包 (Complete Knapsack)

## 1. 概念与定义

完全背包问题是背包问题的一种变体。与0-1背包不同的是，**每种物品可以选择无限次**。只要背包还有空间，同一种物品可以被选取任意多次。

完全背包的核心特点：
- 有 n 种物品，每种物品重量为 w[i]，价值为 v[i]
- 每种物品有**无限个**
- 背包容量为 W
- 目标：在不超过背包容量的前提下，使物品总价值最大

完全背包的经典应用：
- 零钱兑换（凑成目标金额的最少硬币数）
- 零钱兑换II（凑成目标金额的组合数）
- 完全平方数（用最少的完全平方数之和表示n）

## 2. 状态定义与转移方程

### 2.1 二维状态

```
dp[i][j] = 前i种物品、容量为j时的最大价值
转移：dp[i][j] = max(dp[i-1][j], dp[i][j-w[i]] + v[i])
                                        ↑ 注意这里是 dp[i][j-w[i]]
```

**与0-1背包的关键区别**：
- 0-1背包：`dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])`
- 完全背包：`dp[i][j] = max(dp[i-1][j], dp[i][j-w[i]] + v[i])`

### 2.2 一维状态（空间优化）

```
dp[j] = max(dp[j], dp[j-w[i]] + v[i])
遍历顺序：正序遍历（从小到大）
```

**为什么完全背包正序、0-1背包逆序？**
- 0-1背包逆序：保证 `dp[j-w[i]]` 是上一轮（未选当前物品）的结果
- 完全背包正序：`dp[j-w[i]]` 是当前轮已更新的结果，允许重复选取

### 2.3 求解类型

```
1. 最大价值：dp[j] = max(dp[j], dp[j-w[i]] + v[i])
2. 方案数（组合）：dp[j] += dp[j-w[i]]
3. 最少数量：dp[j] = min(dp[j], dp[j-w[i]] + 1)
4. 可行性：dp[j] = dp[j] or dp[j-w[i]]
```

## 3. 算法实现

### 3.1 完全背包 — 最大价值

```python
def complete_knapsack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(weights[i], capacity + 1):  # 正序
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

### 3.2 零钱兑换（LeetCode 322）

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 3.3 零钱兑换 II（LeetCode 518）

```python
def change(amount, coins):
    """组合数：外层遍历物品"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] += dp[j - coin]
    return dp[amount]
```

**排列数（内外循环交换）**：

```python
def change_permutation(amount, coins):
    """排列数：外层遍历容量"""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for j in range(1, amount + 1):
        for coin in coins:
            if j >= coin:
                dp[j] += dp[j - coin]
    return dp[amount]
```

### 3.4 完全平方数（LeetCode 279）

```python
def numSquares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    i = 1
    while i * i <= n:
        square = i * i
        for j in range(square, n + 1):
            dp[j] = min(dp[j], dp[j - square] + 1)
        i += 1
    return dp[n]
```

### 3.5 单词拆分（LeetCode 139）

```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for word in word_set:
            if i >= len(word) and dp[i - len(word)] and s[i - len(word):i] == word:
                dp[i] = True
                break
    return dp[n]
```

### 3.6 C++ 实现

```cpp
int completeKnapsack(vector<int>& w, vector<int>& v, int W) {
    int n = w.size();
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < n; i++)
        for (int j = w[i]; j <= W; j++)  // 正序
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
    return dp[W];
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 完全背包 | O(nW) | O(W) |
| 零钱兑换 | O(n*amount) | O(amount) |
| 完全平方数 | O(n*sqrt(n)) | O(n) |

## 5. 典型例题

### 例题1：组合总和 IV（LeetCode 377）

```python
def combinationSum4(nums, target):
    """排列数"""
    dp = [0] * (target + 1)
    dp[0] = 1
    for j in range(1, target + 1):
        for num in nums:
            if j >= num:
                dp[j] += dp[j - num]
    return dp[target]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **遍历顺序搞错**：完全背包必须正序，0-1背包必须逆序
2. **组合 vs 排列**：
   - 组合：外层枚举物品，内层枚举容量
   - 排列：外层枚举容量，内层枚举物品
3. **dp[0] 初始化**：根据题意确定 dp[0] = 0 或 dp[0] = 1

### 6.2 与0-1背包对比

| 特征 | 0-1背包 | 完全背包 |
|------|--------|---------|
| 每个物品可选次数 | 最多1次 | 无限次 |
| 遍历顺序（一维） | 容量逆序 | 容量正序 |
| 转移来源 | `dp[i-1][j-w[i]]` | `dp[i][j-w[i]]` |

### 6.3 判断完全背包的标志

- 题目说"每种物品有无限个"或"可以重复使用"
- 硬币、骰子等可以重复选择的场景
- 求"最少/最多需要多少个"
