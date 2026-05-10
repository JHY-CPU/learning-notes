# 划分DP (Partition DP)

## 1. 概念与定义

划分DP是将一个序列划分为若干部分，使某个目标最优的动态规划问题。核心思想是枚举划分点，将问题分解为子问题。

常见应用：
- 字符串分割（用字典中的单词分割）
- 整数划分（将n表示为若干正整数之和）
- 分割数组的最大值最小化
- 回文分割

## 2. 状态定义与转移方程

### 2.1 字符串分割

```
dp[i] = s[0..i-1] 是否可以被字典中的单词分割
dp[i] = any(dp[j] and s[j:i] in dict) for j in [0, i)
dp[0] = True
```

### 2.2 分割数组最大值最小化（LeetCode 410）

```
dp[i][j] = 将前j个元素分成i组，各组和的最大值的最小值
dp[i][j] = min(max(dp[i-1][k], sum(k+1..j))) for k in [i-2, j-1)
等价于：二分答案 + 贪心验证
```

### 2.3 整数划分

```
dp[i][j] = 将i划分为j个正整数的方案数
dp[i][j] = dp[i-1][j-1] + dp[i-j][j]
即：至少有一个1（划掉一个1和一个位置）vs 没有1（每个数都>=2，全部减1）
```

## 3. 算法实现

### 3.1 单词拆分（LeetCode 139）

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

### 3.2 单词拆分II（LeetCode 140）

```python
def wordBreakII(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]

    for i in range(1, n + 1):
        for j in range(i):
            word = s[j:i]
            if word in word_set:
                for prev in dp[j]:
                    dp[i].append(prev + [word])

    return [' '.join(words) for words in dp[n]]
```

### 3.3 分割数组的最大值最小化（LeetCode 410）

```python
def splitArray(nums, k):
    """二分答案 + 贪心验证"""
    def can_split(max_sum):
        count = 1
        curr_sum = 0
        for num in nums:
            if curr_sum + num > max_sum:
                count += 1
                curr_sum = num
                if count > k:
                    return False
            else:
                curr_sum += num
        return True

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

### 3.4 整数划分计数

```python
def integerPartition(n, k):
    """将n分成恰好k个正整数的方案数"""
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i-1][j-1] + dp[i-j][j]

    return dp[n][k]
```

### 3.5 完全背包视角的整数划分

```python
def countPartitions(n):
    """将n分成若干正整数之和的方案数（顺序无关）"""
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):  # 物品：1,2,3,...,n
        for j in range(i, n + 1):
            dp[j] += dp[j - i]
    return dp[n]
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 单词拆分 | O(n²) | O(n) |
| 单词拆分II | O(n³) | O(n³) |
| 分割数组 | O(nlog(sum)) | O(1) |
| 整数划分 | O(nk) | O(nk) |

## 5. 典型例题

### 例题1：将数组分成k个连续子数组（LeetCode 698）

```python
def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    nums.sort(reverse=True)

    if nums[0] > target:
        return False

    groups = [0] * k

    def backtrack(idx):
        if idx == len(nums):
            return all(g == target for g in groups)
        for i in range(k):
            if groups[i] + nums[idx] <= target:
                groups[i] += nums[idx]
                if backtrack(idx + 1):
                    return True
                groups[i] -= nums[idx]
            if groups[i] == 0:
                break  # 剪枝
        return False

    return backtrack(0)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **单词拆分超时**：需要先检查能否拆分（139题），再求方案（140题）
2. **划分时的边界**：空集是否合法
3. **最大值最小化**：常用二分答案 + 贪心

### 6.2 划分DP的一般形式

```
dp[i] = min/max(dp[j] + cost(j+1, i)) for j in [0, i)
或
dp[i][k] = min/max(dp[j][k-1] + cost(j+1, i)) for j in [0, i)
```
