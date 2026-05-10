# 模拟面试-DP (Mock Interview - DP)

## 一、面试流程模拟

**时间：** 45分钟
**重点：** 状态定义、状态转移、空间优化

---

## 二、题目1：零钱兑换 (LeetCode 322, Medium, 15分钟)

### 题目描述

给定不同面额的硬币 `coins` 和总金额 `amount`，计算凑成总金额所需的最少硬币数。不能凑成返回 -1。

### 面试过程

**候选人：**
"完全背包DP。`dp[i]` 表示凑成金额 `i` 的最少硬币数。

状态转移：`dp[i] = min(dp[i], dp[i-c] + 1)` 对每个硬币面额 c。"

### 代码

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] != float('inf'):
                dp[i] = min(dp[i], dp[i - c] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

**面试官追问：** "怎么记录具体方案？"
"额外维护一个 `choice[i]` 记录到达状态 i 时选了哪个硬币，最后从 amount 倒推即可。"

**复杂度：** 时间 $O(amount \times n)$，空间 $O(amount)$。

---

## 三、题目2：最长公共子序列 (LeetCode 1143, Medium, 15分钟)

### 面试过程

**候选人：**
"`dp[i][j]` 表示 `text1[:i]` 和 `text2[:j]` 的 LCS 长度。

字符相同时：`dp[i][j] = dp[i-1][j-1] + 1`
不同时：`dp[i][j] = max(dp[i-1][j], dp[i][j-1])`"

### 代码

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

**面试官追问：** "空间优化？"
"可以只用两行（滚动数组），进一步优化到一行需要从右往左更新。"

---

## 四、题目3：买卖股票的最佳时机含冷冻期 (LeetCode 309, Medium, 15分钟)

### 面试过程

**候选人：**
"定义三个状态：
- `hold[i]`：第i天持有股票的最大收益
- `sold[i]`：第i天刚卖出的最大收益
- `rest[i]`：第i天不持有也不卖出的最大收益"

### 代码

```python
def max_profit(prices):
    if len(prices) < 2:
        return 0

    hold = -prices[0]  # 持有
    sold = 0           # 刚卖出
    rest = 0           # 休息

    for i in range(1, len(prices)):
        prev_sold = sold
        sold = hold + prices[i]
        hold = max(hold, rest - prices[i])
        rest = max(rest, prev_sold)

    return max(sold, rest)
```

**面试官追问：** "如果限制最多交易k次呢？"
"状态变成 `dp[i][k][0/1]`，i是天数，k是剩余交易次数，0/1表示是否持有。空间可以优化到 O(k)。"

---

## 五、评分要点

1. **状态定义** — 能否准确描述dp数组的含义
2. **转移方程** — 能否推导出正确的递推公式
3. **初始化** — 边界条件是否正确
4. **优化意识** — 空间优化、滚动数组
5. **变种处理** — 能否应对参数变化
