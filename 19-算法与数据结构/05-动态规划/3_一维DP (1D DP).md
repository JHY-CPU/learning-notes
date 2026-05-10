# 一维DP (1D DP)

## 1. 概念与定义

一维DP是指状态只需要一个维度即可描述的动态规划问题。通常用 `dp[i]` 表示某种含义，其中 `i` 是一个整数索引。一维DP是最基础的DP形式。

一维DP的常见模式：
- **线性DP**：沿数组/序列方向递推，如最大子数组和
- **计数DP**：计算方案总数，如爬楼梯、解码方法
- **最值DP**：求最优值，如打家劫舍、最长递增子序列
- **可行性DP**：判断是否可行，如单词拆分

## 2. 状态定义与转移方程

### 2.1 斐波那契数列
```
dp[i] = dp[i-1] + dp[i-2]
dp[0] = 0, dp[1] = 1
```

### 2.2 爬楼梯
```
dp[i] = dp[i-1] + dp[i-2]
dp[0] = 1, dp[1] = 1
```

### 2.3 打家劫舍
```
dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
dp[0] = 0, dp[1] = nums[0]
```

### 2.4 最大子数组和
```
dp[i] = max(nums[i], dp[i-1] + nums[i])
dp[0] = nums[0]
答案：max(dp)
```

### 2.5 最长递增子序列
```
dp[i] = max(dp[j] + 1) for j < i and nums[j] < nums[i]
dp[i] = 1 初始
答案：max(dp)
```

## 3. 算法实现

### 3.1 爬楼梯（LeetCode 70）

```python
def climbStairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    return b

# 通用版：每次可以爬1~m步
def climbStairs_general(n, m):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            dp[i] += dp[i - j]
    return dp[n]
```

### 3.2 打家劫舍系列

```python
# LeetCode 198
def rob(nums):
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[-1]

# LeetCode 213 环形
def rob_II(nums):
    if len(nums) == 1:
        return nums[0]
    def rob_range(start, end):
        prev2 = nums[start]
        prev1 = max(nums[start], nums[start + 1])
        for i in range(start + 2, end + 1):
            prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
        return prev1
    return max(rob_range(0, len(nums) - 2), rob_range(1, len(nums) - 1))
```

### 3.3 最大子数组和（LeetCode 53）

```python
def maxSubArray(nums):
    """Kadane算法"""
    max_sum = curr_sum = nums[0]
    for i in range(1, len(nums)):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

### 3.4 解码方法（LeetCode 91）

```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]
    return dp[n]
```

### 3.5 C++ 实现

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    vector<int> dp(n, 0);
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    for (int i = 2; i < n; i++)
        dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
    return dp[n-1];
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 | 优化后空间 |
|------|-----------|-----------|-----------|
| 斐波那契 | O(n) | O(n) | O(1) |
| 爬楼梯 | O(n) | O(n) | O(1) |
| 打家劫舍 | O(n) | O(n) | O(1) |
| 最大子数组和 | O(n) | O(n) | O(1) |
| LIS | O(n²) | O(n) | O(nlogn) |
| 零钱兑换 | O(n*amount) | O(amount) | - |

## 5. 典型例题

### 例题1：删除并获得点数（LeetCode 740）

```python
def deleteAndEarn(nums):
    from collections import Counter
    count = Counter(nums)
    max_val = max(nums)
    points = [0] * (max_val + 1)
    for num, freq in count.items():
        points[num] = num * freq
    dp = [0] * (max_val + 1)
    dp[1] = points[1]
    for i in range(2, max_val + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + points[i])
    return dp[max_val]
```

### 例题2：单词拆分（LeetCode 139）

```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **索引偏移**：`dp[i]` 代表前i个元素时，`nums` 的索引是 `i-1`
2. **边界条件**：空数组、单元素数组需要特殊处理
3. **整数溢出**：大数计算需要取模
4. **遍历顺序**：确保子问题先于当前问题计算

### 6.2 优化方向

1. **空间优化**：只依赖最近两个状态时用两个变量
2. **前缀和优化**：区间求和从O(n)优化到O(1)
3. **单调队列优化**：滑动窗口类问题
4. **二分优化**：LIS从O(n²)优化到O(nlogn)

### 6.3 解题模板

```python
def solve_1d_dp(arr):
    n = len(arr)
    if n == 0: return 0
    dp = [0] * n
    dp[0] = 基础值
    for i in range(1, n):
        dp[i] = 通过dp[0..i-1]计算
    return dp[n - 1]  # 或 max(dp) / dp[n]
```
