# 状态压缩 (State Compression)

## 1. 概念与定义

状态压缩是用**位运算**将集合/状态压缩为整数的技巧。当问题的状态可以用"选择/不选择"、"使用/未使用"来描述时，可以用二进制数的每一位表示一个元素的状态。

状态压缩DP（状压DP）适用于：
- 元素数量不超过约20
- 状态可以用集合表示
- 需要枚举所有子集

## 2. 常用位运算

```python
# 检查第i位
(mask >> i) & 1

# 设置第i位为1
mask | (1 << i)

# 设置第i位为0
mask & ~(1 << i)

# 翻转第i位
mask ^ (1 << i)

# 枚举所有子集
sub = mask
while sub:
    # 处理子集 sub
    sub = (sub - 1) & mask

# lowbit
lowbit = mask & (-mask)

# 统计1的个数
bin(mask).count('1')
# 或 Python 3.10+
mask.bit_count()
```

## 3. 算法实现

### 3.1 旅行商问题（TSP）

```python
def tsp(dist):
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]

    for i in range(n):
        dp[1 << i][i] = 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] == INF:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    return min(dp[(1 << n) - 1])
```

### 3.2 最短Hamilton路径

```python
def shortestHamiltonPath(graph):
    n = len(graph)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 从节点0出发

    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])

    return dp[(1 << n) - 1][n - 1]
```

### 3.3 集合划分

```python
def minSubsetDifference(nums):
    """将数组分成两组，使差值最小"""
    total = sum(nums)
    n = len(nums)
    result = float('inf')

    for mask in range(1 << n):
        s = sum(nums[i] for i in range(n) if (mask >> i) & 1)
        result = min(result, abs(total - 2 * s))

    return result

# DP优化：当n较大时用DP
def minSubsetDifference_dp(nums, target):
    """能否从nums中选出和为target的子集"""
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| TSP | O(2^n * n²) | O(2^n * n) |
| 集合划分（暴力） | O(2^n) | O(1) |
| 集合划分（DP） | O(n * sum) | O(sum) |

## 5. 典型例题

### 例题1：公平分配饼干（LeetCode 2305）

```python
def distributeCookies(cookies, k):
    n = len(cookies)
    total = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if (mask >> i) & 1:
                total[mask] += cookies[i]

    dp = [[float('inf')] * (k + 1) for _ in range(1 << n)]
    dp[0][0] = 0

    for mask in range(1 << n):
        for j in range(k):
            if dp[mask][j] == float('inf'):
                continue
            remain = ((1 << n) - 1) ^ mask
            sub = remain
            while sub:
                dp[mask | sub][j + 1] = min(dp[mask | sub][j + 1], max(dp[mask][j], total[sub]))
                sub = (sub - 1) & remain

    return dp[(1 << n) - 1][k]
```

## 6. 常见陷阱

### 6.1 陷阱

1. **n太大**：n > 25 时 2^n 太大，无法使用
2. **位运算优先级**：`==` 优先级高于 `&`，需要加括号
3. **枚举子集遗漏**：空集也要考虑
4. **内存**：2^20 * 20 ≈ 20M，可能MLE

### 6.2 适用判断

- n <= 20：可以用状态压缩
- n <= 15：状态压缩很轻松
- n > 25：考虑其他方法
