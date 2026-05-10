# 股票买卖DP (Stock DP)

## 1. 概念与定义

股票买卖问题是LeetCode上经典的DP系列问题。核心是在给定每日股票价格序列中，通过买卖操作获得最大利润。不同题目有不同的约束：
- 最多交易 k 次
- 是否有冷冻期
- 是否有手续费
- 是否同时持有多只股票

核心思想：**状态机DP** — 用不同的状态表示"持有/未持有"、"交易次数"等。

## 2. 状态定义与转移方程

### 2.1 只能买卖一次（LeetCode 121）

```
dp[i] = 第i天之前（含第i天）的最低价格
result = max(prices[i] - dp[i])
等价于：维护最低价格，每天计算利润
```

### 2.2 无限次交易（LeetCode 122）

```
贪心：只要明天比今天贵就买卖
dp[i][0] = 第i天不持有股票的最大利润
dp[i][1] = 第i天持有股票的最大利润
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
```

### 2.3 最多k次交易（LeetCode 188）

```
dp[i][j][0/1] = 第i天、已交易j次、不持有/持有股票的最大利润
dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])
```

### 2.4 含冷冻期（LeetCode 309）

```
dp[i][0] = 第i天持有股票
dp[i][1] = 第i天刚卖出（冷冻期）
dp[i][2] = 第i天不持有且不在冷冻期
```

### 2.5 含手续费（LeetCode 714）

```
dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i] - fee)
dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
```

## 3. 算法实现

### 3.1 只能买卖一次（LeetCode 121）

```python
def maxProfit_one(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

### 3.2 无限次交易（LeetCode 122）

```python
def maxProfit_unlimited(prices):
    dp0, dp1 = 0, -prices[0]
    for i in range(1, len(prices)):
        dp0, dp1 = max(dp0, dp1 + prices[i]), max(dp1, dp0 - prices[i])
    return dp0
```

### 3.3 最多两次交易（LeetCode 123）

```python
def maxProfit_k2(prices):
    n = len(prices)
    # dp[j][0/1]: 交易j次、不持有/持有
    dp = [[0, -prices[0]] for _ in range(3)]  # j = 0, 1, 2
    for i in range(1, n):
        for j in range(2, 0, -1):  # 逆序更新
            dp[j][0] = max(dp[j][0], dp[j][1] + prices[i])
            dp[j][1] = max(dp[j][1], dp[j-1][0] - prices[i])
    return dp[2][0]
```

### 3.4 最多k次交易（LeetCode 188）

```python
def maxProfit_k(k, prices):
    if not prices:
        return 0
    if k >= len(prices) // 2:
        # k足够大，等价于无限次交易
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))

    n = len(prices)
    dp = [[0, -prices[0]] for _ in range(k + 1)]
    for i in range(1, n):
        for j in range(k, 0, -1):
            dp[j][0] = max(dp[j][0], dp[j][1] + prices[i])
            dp[j][1] = max(dp[j][1], dp[j-1][0] - prices[i])
    return dp[k][0]
```

### 3.5 含冷冻期（LeetCode 309）

```python
def maxProfit_cooldown(prices):
    n = len(prices)
    if n <= 1:
        return 0
    hold = -prices[0]     # 持有
    sold = 0              # 刚卖出（冷冻期）
    rest = 0              # 不持有，不在冷冻期

    for i in range(1, n):
        prev_sold = sold
        sold = hold + prices[i]
        hold = max(hold, rest - prices[i])
        rest = max(rest, prev_sold)

    return max(sold, rest)
```

### 3.6 含手续费（LeetCode 714）

```python
def maxProfit_fee(prices, fee):
    dp0, dp1 = 0, -prices[0]
    for i in range(1, len(prices)):
        dp0, dp1 = max(dp0, dp1 + prices[i] - fee), max(dp1, dp0 - prices[i])
    return dp0
```

### 3.7 C++ 实现

```cpp
// 无限次交易
int maxProfit(vector<int>& prices) {
    int dp0 = 0, dp1 = -prices[0];
    for (int i = 1; i < prices.size(); i++) {
        int new_dp0 = max(dp0, dp1 + prices[i]);
        int new_dp1 = max(dp1, dp0 - prices[i]);
        dp0 = new_dp0;
        dp1 = new_dp1;
    }
    return dp0;
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 一次交易 | O(n) | O(1) |
| 无限次交易 | O(n) | O(1) |
| k次交易 | O(nk) | O(k) |
| 含冷冻期 | O(n) | O(1) |
| 含手续费 | O(n) | O(1) |

## 5. 典型例题

### LeetCode 系列总结

| 题号 | 题名 | 限制 | 核心 |
|------|------|------|------|
| 121 | 买卖股票最佳时机 | 1次 | 维护最低价 |
| 122 | 买卖股票II | 无限次 | 贪心或DP |
| 123 | 买卖股票III | 2次 | 二维交易次数 |
| 188 | 买卖股票IV | k次 | 通用k次DP |
| 309 | 含冷冻期 | 冷冻期 | 三状态 |
| 714 | 含手续费 | 手续费 | 减去fee |

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **交易次数k太大**：k >= n/2 时等价于无限次，直接贪心
2. **逆序更新**：k次交易中j需要逆序，避免同一轮覆盖
3. **初始值**：dp[1] = -prices[0]（第一天买入）
4. **冷冻期**：注意卖出后第二天才能买

### 6.2 状态机思想

```
持有 → 卖出 → 冷冻期 → 可以买入
  ↑__________________________|

每种状态都是一个DP变量
转移时注意状态之间的合法转换
```
