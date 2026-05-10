# DP与贪心对比 (DP vs Greedy)

## 1. 概念与定义

贪心算法和动态规划都是求解最优化问题的方法，但它们的适用条件和工作原理不同。

**贪心算法**：每一步都做出局部最优选择，期望最终得到全局最优解。
**动态规划**：考虑所有可能的选择，通过子问题的最优解推导原问题的最优解。

关键区别：贪心只考虑当前最优，DP考虑所有子问题的最优。

## 2. 对比分析

### 2.1 核心区别

| 特性 | 贪心 | 动态规划 |
|------|------|---------|
| 选择策略 | 局部最优 | 全局最优 |
| 子问题 | 不重叠 | 重叠 |
| 最优子结构 | 必须有 | 必须有 |
| 贪心选择性质 | 必须有 | 不需要 |
| 时间复杂度 | 通常O(n) | 通常O(n²)或更高 |
| 空间复杂度 | 通常O(1) | 通常O(n)或更高 |

### 2.2 什么时候用贪心

需要满足两个条件：
1. **最优子结构**：问题的最优解包含子问题的最优解
2. **贪心选择性质**：通过局部最优选择可以得到全局最优解

贪心选择性质比最优子结构更强 — 不仅子问题的最优能组合成全局最优，而且每一步贪心选的就是全局最优的一部分。

## 3. 典型对比

### 3.1 零钱兑换

```
贪心：每次选最大面额的硬币
DP：考虑所有可能的硬币组合
```

**贪心失败的例子**：面额为 [1, 3, 4]，凑6元
- 贪心：4 + 1 + 1 = 3枚
- 最优：3 + 3 = 2枚

```python
# 贪心（仅适用于特定面额如[1,5,10,25]）
def coinChange_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1

# DP（通用）
def coinChange_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 3.2 背包问题

```
分数背包：贪心（按性价比排序）
0-1背包：DP（不能贪心）
```

```python
# 分数背包 — 贪心有效
def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x[1]/x[0], reverse=True)  # 按性价比
    total = 0
    for w, v in items:
        if capacity >= w:
            total += v
            capacity -= w
        else:
            total += v * (capacity / w)
            break
    return total

# 0-1背包 — 必须DP
def knapsack_01(items, capacity):
    dp = [0] * (capacity + 1)
    for w, v in items:
        for j in range(capacity, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[capacity]
```

### 3.3 区间调度

```
活动选择问题：贪心有效
区间覆盖问题：可能需要DP
```

```python
# 活动选择 — 贪心
def maxActivities(activities):
    activities.sort(key=lambda x: x[1])  # 按结束时间排序
    count = 0
    end = 0
    for s, e in activities:
        if s >= end:
            count += 1
            end = e
    return count

# 最大区间不重叠子集 — 类似贪心
```

## 4. 如何判断用哪个

### 4.1 决策流程

```
1. 尝试贪心：画出贪心选择的过程
2. 反例检查：能否构造一个反例使贪心失败？
3. 如果贪心正确 → 使用贪心（更高效）
4. 如果贪心失败 → 使用DP
```

### 4.2 贪心通常有效的场景

- 区间调度（按结束时间排序）
- 哈夫曼编码（每次合并最小的两个）
- 分数背包（按性价比排序）
- 最小生成树（Prim/Kruskal）
- 最短路径（Dijkstra）

### 4.3 贪心通常失败的场景

- 0-1背包（物品不可分割）
- 最长递增子序列（不能只看当前最优）
- 零钱兑换（任意面额组合）
- 最短路径（有负权边时Dijkstra失败）

## 5. 证明贪心正确性

### 5.1 交换论证

假设存在最优解与贪心解不同，通过交换论证证明贪心解不比最优解差。

### 5.2 归纳法

1. 证明第一步贪心选择是正确的
2. 假设前k步贪心选择是正确的
3. 证明第k+1步贪心选择也是正确的

## 6. 常见陷阱

### 6.1 陷阱

1. **直觉错误**：看似能贪心的问题实际需要DP
2. **特殊数据**：贪心只对特定数据有效
3. **排序键选错**：贪心需要正确的排序策略

### 6.2 实战建议

1. 先尝试贪心，因为代码更简单
2. 用小数据测试贪心是否正确
3. 如果不确定，用DP（DP总是正确的，只是可能不够高效）
