# 动态规划专题 (DP Problems)

## 一、DP核心思想

### 1.1 三大要素

1. **最优子结构：** 最优解包含子问题的最优解
2. **重叠子问题：** 子问题被多次求解
3. **无后效性：** 状态一旦确定，不受后续决策影响

### 1.2 解题五步法

```
1. 定义状态 — dp[i] 或 dp[i][j] 代表什么
2. 写状态转移方程 — dp[i] 如何从更小的状态推导
3. 确定初始条件 — dp[0] 等基础值
4. 确定遍历顺序 — 从前往后还是从后往前
5. 返回结果 — dp[n] 或 dp[m][n]
```

---

## 二、经典DP类型

### 2.1 01背包

$n$ 件物品，每件最多选一次，背包容量 $W$：

```python
def knapsack_01(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for j in range(W, w - 1, -1):  # 逆序！
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

### 2.2 完全背包

每件物品可选无限次：

```python
def knapsack_complete(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for j in range(w, W + 1):  # 正序！
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[W]
```

### 2.3 最长递增子序列 (LIS)

```python
# O(n log n) 贪心+二分
from bisect import bisect_left

def length_of_lis(nums):
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
```

### 2.4 最长公共子序列 (LCS)

```python
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### 2.5 编辑距离 (LeetCode 72)

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # 删除
                                   dp[i][j-1],    # 插入
                                   dp[i-1][j-1])  # 替换
    return dp[m][n]
```

---

## 三、区间DP

### 3.1 戳气球 (LeetCode 312)

```python
def max_coins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left + 1, right):
                coins = nums[left] * nums[k] * nums[right]
                dp[left][right] = max(dp[left][right],
                    dp[left][k] + dp[k][right] + coins)

    return dp[0][n-1]
```

---

## 四、股票交易DP

### 4.1 含冷冻期 (LeetCode 309)

```python
def max_profit_with_cooldown(prices):
    n = len(prices)
    if n < 2: return 0
    hold, sold, rest = -prices[0], 0, 0
    for i in range(1, n):
        prev_sold = sold
        sold = hold + prices[i]
        hold = max(hold, rest - prices[i])
        rest = max(rest, prev_sold)
    return max(sold, rest)
```

---

## 五、状态压缩DP

### 5.1 TSP旅行商问题

```python
def tsp(dist):
    n = len(dist)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 从城市0出发

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)): continue
            for v in range(n):
                if mask & (1 << v): continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v],
                                      dp[mask][u] + dist[u][v])

    return min(dp[(1<<n)-1][i] + dist[i][0] for i in range(n))
```

---

## 六、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 最长递增子序列
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;
    for (int x : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else *it = x;
    }
    return tails.size();
}

// 编辑距离
int minDistance(string w1, string w2) {
    int m = w1.size(), n = w2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (w1[i-1] == w2[j-1]) dp[i][j] = dp[i-1][j-1];
            else dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
    return dp[m][n];
}
```

---

## 七、复杂度分析

| 问题 | 时间 | 空间 |
|------|------|------|
| 01背包 | $O(nW)$ | $O(W)$ |
| LIS $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| LIS $O(n\log n)$ | $O(n\log n)$ | $O(n)$ |
| LCS | $O(mn)$ | $O(mn)$ |
| 编辑距离 | $O(mn)$ | $O(mn)$ |
| 区间DP | $O(n^3)$ | $O(n^2)$ |
| 状态压缩 | $O(2^n \cdot n)$ | $O(2^n \cdot n)$ |

---

## 八、面试高频题

1. **LeetCode 300：** 最长递增子序列
2. **LeetCode 1143：** 最长公共子序列
3. **LeetCode 322：** 零钱兑换
4. **LeetCode 518：** 零钱兑换II
5. **LeetCode 72：** 编辑距离
6. **LeetCode 139：** 单词拆分
7. **LeetCode 312：** 戳气球
8. **LeetCode 309：** 含冷冻期的股票买卖
9. **LeetCode 62/63/64：** 不同路径系列
10. **LeetCode 5：** 最长回文子串
