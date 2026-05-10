# 状态转移方程 (State Transition)

## 1. 概念与定义

状态转移方程是动态规划的核心，它描述了子问题之间如何相互推导。写对状态转移方程，就解决了DP问题的80%。状态转移方程回答的问题是：**当前状态可以由哪些子状态推导出来？**

一个完整的DP需要明确以下四个要素：
1. **状态定义**：`dp[i]` 的含义是什么
2. **转移方程**：`dp[i]` 如何由更小的子问题推导
3. **初始条件**：最小的子问题的解
4. **遍历顺序**：按什么顺序计算保证子问题已求解

## 2. 状态定义与转移方程

### 2.1 状态定义的方法

#### 方法一：按结尾位置定义

`dp[i]` 表示以第 i 个元素结尾的某个最优值。
- 最大子数组和：`dp[i] = max(nums[i], dp[i-1] + nums[i])`
- 最长递增子序列：`dp[i] = max(dp[j] + 1)` for all `j < i` and `nums[j] < nums[i]`

#### 方法二：按前缀定义

`dp[i]` 表示前 i 个元素的某个最优值。
- 爬楼梯：`dp[i] = dp[i-1] + dp[i-2]`
- 打家劫舍：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

#### 方法三：按区间定义

`dp[i][j]` 表示区间 [i, j] 上的最优值。
- 石子合并：`dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost(i,j))`

#### 方法四：多维状态

`dp[i][j]` 用多个维度描述状态。
- 0-1背包：`dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])`

### 2.2 经典转移方程汇总

| 问题 | 状态定义 | 转移方程 |
|------|---------|---------|
| 斐波那契 | `dp[i]`第i项值 | `dp[i] = dp[i-1] + dp[i-2]` |
| 最大子数组和 | `dp[i]`以i结尾的最大和 | `dp[i] = max(nums[i], dp[i-1]+nums[i])` |
| 打家劫舍 | `dp[i]`前i间最大金额 | `dp[i] = max(dp[i-1], dp[i-2]+nums[i])` |
| 0-1背包 | `dp[i][j]`前i件容量j最大价值 | `dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]]+v[i])` |
| LCS | `dp[i][j]`两串前i/j的LCS长度 | 见LCS专题 |
| 编辑距离 | `dp[i][j]`最少操作数 | 见编辑距离专题 |

## 3. 算法实现

### 3.1 推导转移方程的系统方法

```python
# 示例：打家劫舍的推导过程
# 1. dp[i] = 前i间房屋能偷的最大金额
# 2. 对于第i间房：偷 或 不偷
#    - 偷：dp[i] = dp[i-2] + nums[i]
#    - 不偷：dp[i] = dp[i-1]
# 3. 取最优：dp[i] = max(dp[i-1], dp[i-2] + nums[i])
# 4. 初始条件：dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
```

### 3.2 最大子数组和（LeetCode 53）

```python
def maxSubArray(nums):
    """
    状态定义：dp[i] = 以nums[i]结尾的最大子数组和
    转移方程：dp[i] = max(nums[i], dp[i-1] + nums[i])
    """
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    result = dp[0]

    for i in range(1, n):
        dp[i] = max(nums[i], dp[i - 1] + nums[i])
        result = max(result, dp[i])

    return result
```

### 3.3 打家劫舍（LeetCode 198）

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[n - 1]
```

### 3.4 乘积最大子数组（LeetCode 152）

```python
def maxProduct(nums):
    """同时维护最大值和最小值（负数乘负数变大）"""
    n = len(nums)
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = nums[0]
    result = nums[0]

    for i in range(1, n):
        max_dp[i] = max(nums[i], max_dp[i-1] * nums[i], min_dp[i-1] * nums[i])
        min_dp[i] = min(nums[i], max_dp[i-1] * nums[i], min_dp[i-1] * nums[i])
        result = max(result, max_dp[i])

    return result
```

## 4. 复杂度分析

```
时间复杂度 = 状态数 × 单个状态的转移代价
空间复杂度 = 状态空间大小
```

- 一维DP：O(n) 个状态，每个状态 O(1) 转移，总复杂度 O(n)
- 二维DP：O(n²) 个状态，总复杂度 O(n²)
- 区间DP：O(n²) 个状态，每个状态 O(n) 转移，总复杂度 O(n³)

## 5. 典型例题

### 例题1：零钱兑换（LeetCode 322）

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 例题2：打家劫舍II（LeetCode 213）环形

```python
def rob(nums):
    if len(nums) == 1:
        return nums[0]
    def rob_range(start, end):
        prev2, prev1 = nums[start], max(nums[start], nums[start + 1])
        for i in range(start + 2, end + 1):
            prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
        return prev1
    return max(rob_range(0, len(nums) - 2), rob_range(1, len(nums) - 1))
```

## 6. 常见陷阱与优化

### 6.1 转移方程常见错误

1. **转移不完整**：漏掉某些情况
2. **方向错误**：转移方向导致循环依赖
3. **边界处理不当**：未考虑空集、单元素等特殊情况
4. **取模问题**：方案数DP忘记取模导致溢出

### 6.2 如何检查正确性

1. **手算小数据**：用 n=1,2,3 手算验证
2. **画递归树**：看是否有遗漏的子问题
3. **验证边界**：确保初始条件正确
4. **对拍**：写暴力解法，与DP解法对比随机数据

### 6.3 常用技巧

1. **添加虚拟边界**：dp[0] 设为特殊值简化边界处理
2. **负无穷/正无穷初始化**：求最大值时初始化为负无穷
3. **维度扩展**：增加一维状态让转移更清晰
4. **前缀和预处理**：区间求和优化为 O(1)
