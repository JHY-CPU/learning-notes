# 背包总结 (Knapsack Summary)

## 1. 概念与定义

背包问题是动态规划中最经典、最重要的问题类型之一。核心是在**有限资源（容量）**约束下，从若干选项中做出**最优选择**。

背包问题的分类：
- **0-1背包**：每种物品最多选1次
- **完全背包**：每种物品可选无限次
- **多重背包**：每种物品有给定数量限制
- **分组背包**：物品分为若干组，每组内最多选1个
- **混合背包**：以上类型的组合

## 2. 状态定义与转移方程

### 2.1 三种背包对比

```
0-1背包：dp[j] = max(dp[j], dp[j-w[i]] + v[i])    遍历：容量逆序
完全背包：dp[j] = max(dp[j], dp[j-w[i]] + v[i])    遍历：容量正序
多重背包：dp[j] = max(dp[j], dp[j-k*w[i]] + k*v[i]) 优化：二进制拆分
```

### 2.2 变体：恰好装满

```python
# 不要求恰好装满
dp = [0] * (W + 1)
# 要求恰好装满
dp = [-float('inf')] * (W + 1)
dp[0] = 0
```

### 2.3 变体：方案数

```python
dp = [0] * (W + 1)
dp[0] = 1
for i in range(n):
    for j in range(W, w[i] - 1, -1):
        dp[j] += dp[j - w[i]]
```

## 3. 算法实现

### 3.1 0-1背包模板

```python
def knapsack_01(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for j in range(capacity, weights[i] - 1, -1):  # 逆序
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

### 3.2 完全背包模板

```python
def knapsack_complete(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for j in range(weights[i], capacity + 1):  # 正序
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

### 3.3 多重背包模板

```python
def knapsack_multiple(weights, values, counts, capacity):
    nw, nv = [], []
    for i in range(len(weights)):
        c = counts[i]
        k = 1
        while k <= c:
            nw.append(k * weights[i])
            nv.append(k * values[i])
            c -= k
            k *= 2
        if c > 0:
            nw.append(c * weights[i])
            nv.append(c * values[i])
    dp = [0] * (capacity + 1)
    for i in range(len(nw)):
        for j in range(capacity, nw[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - nw[i]] + nv[i])
    return dp[capacity]
```

### 3.4 分组背包模板

```python
def knapsack_grouped(groups, capacity):
    dp = [0] * (capacity + 1)
    for group in groups:
        for j in range(capacity, -1, -1):
            for w, v in group:
                if j >= w:
                    dp[j] = max(dp[j], dp[j - w] + v)
    return dp[capacity]
```

### 3.5 C++ 模板

```cpp
// 0-1背包
for (int i = 0; i < n; i++)
    for (int j = W; j >= w[i]; j--)
        dp[j] = max(dp[j], dp[j-w[i]] + v[i]);

// 完全背包
for (int i = 0; i < n; i++)
    for (int j = w[i]; j <= W; j++)
        dp[j] = max(dp[j], dp[j-w[i]] + v[i]);
```

## 4. 复杂度分析

| 背包类型 | 时间复杂度 | 空间复杂度 | 关键优化 |
|---------|-----------|-----------|---------|
| 0-1背包 | O(nW) | O(W) | 一维+逆序 |
| 完全背包 | O(nW) | O(W) | 一维+正序 |
| 多重背包 | O(nW*log(c)) | O(W) | 二进制拆分 |
| 分组背包 | O(nW) | O(W) | 组内逆序 |

## 5. 典型例题

### 例题1：目标和（LeetCode 494）

```python
def findTargetSumWays(nums, target):
    """
    正数集合P，负数集合N，P - N = target，P + N = sum
    => P = (target + sum) / 2
    问题变为：0-1背包求方案数
    """
    total = sum(nums)
    if (target + total) % 2 != 0 or abs(target) > total:
        return 0
    P = (target + total) // 2
    dp = [0] * (P + 1)
    dp[0] = 1
    for num in nums:
        for j in range(P, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[P]
```

### 例题2：一和零（LeetCode 474）

```python
def findMaxForm(strs, m, n):
    """二维背包"""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        zeros = s.count('0')
        ones = s.count('1')
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]
```

### 例题3：最后一块石头的重量II（LeetCode 1049）

```python
def lastStoneWeightII(stones):
    """分成两堆使差值最小 = 0-1背包"""
    total = sum(stones)
    target = total // 2
    dp = [0] * (target + 1)
    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = max(dp[j], dp[j - stone] + stone)
    return total - 2 * dp[target]
```

## 6. 常见陷阱与优化

### 6.1 如何判断背包类型

| 题目特征 | 背包类型 |
|---------|---------|
| 每种最多选一次 | 0-1背包 |
| 每种可选多次 | 完全背包 |
| 每种有数量限制 | 多重背包 |
| 分组选择 | 分组背包 |

### 6.2 遍历顺序口诀

```
0-1背包：逆序（倒着走，防止重复选取）
完全背包：正序（顺着走，允许重复选取）
组合：先物品后容量
排列：先容量后物品
```

### 6.3 常见背包问题转化

1. **能否凑出目标值**：可行性背包
2. **凑出目标值的方案数**：计数背包
3. **凑出目标值的最少物品数**：最小值背包
4. **分割等和子集**：0-1背包
5. **字符串能否被拆分**：完全背包
