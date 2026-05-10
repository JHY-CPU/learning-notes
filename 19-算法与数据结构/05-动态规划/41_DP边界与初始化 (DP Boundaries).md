# DP边界初始化 (DP Boundaries)

## 1. 概念与定义

DP的边界初始化是确保动态规划正确性的关键。错误的边界初始化会导致整个DP计算错误。边界条件包括：
- **初始状态**：最小的子问题的解
- **遍历起点**：从哪个状态开始递推
- **越界保护**：避免数组越界访问

## 2. 常见边界模式

### 2.1 一维DP的边界

```python
# 斐波那契
dp[0] = 0; dp[1] = 1

# 爬楼梯
dp[0] = 1; dp[1] = 1  # 或 dp[1]=1; dp[2]=2

# 打家劫舍
dp[0] = nums[0]; dp[1] = max(nums[0], nums[1])

# 最大子数组和
dp[0] = nums[0]  # 或 max_sum = curr = nums[0]
```

### 2.2 二维DP的边界

```python
# 网格路径
dp[0][0] = grid[0][0]
for i in range(1, m): dp[i][0] = dp[i-1][0] + grid[i][0]
for j in range(1, n): dp[0][j] = dp[0][j-1] + grid[0][j]

# LCS
dp[i][0] = 0  # 空串
dp[0][j] = 0

# 编辑距离
dp[i][0] = i  # 删除i个字符
dp[0][j] = j  # 插入j个字符
```

### 2.3 背包DP的边界

```python
# 最大价值
dp = [0] * (W + 1)

# 恰好装满
dp = [-float('inf')] * (W + 1)
dp[0] = 0

# 方案数
dp = [0] * (W + 1)
dp[0] = 1

# 最少物品数
dp = [float('inf')] * (W + 1)
dp[0] = 0
```

### 2.4 区间DP的边界

```python
# 单个元素
for i in range(n):
    dp[i][i] = 0  # 或 1，根据题意

# 相邻元素
for i in range(n-1):
    dp[i][i+1] = 计算相邻的代价
```

## 3. 算法实现

### 3.1 边界处理示例

```python
def example_with_boundary(nums):
    n = len(nums)
    if n == 0:
        return 特殊值  # 空数组
    if n == 1:
        return nums[0]  # 单元素

    dp = [0] * n
    dp[0] = nums[0]  # 边界1
    dp[1] = max(nums[0], nums[1])  # 边界2

    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[n-1]
```

### 3.2 虚拟边界技巧

```python
def with_dummy_boundary(nums):
    n = len(nums)
    # 添加虚拟边界，简化代码
    dp = [0] * (n + 1)
    dp[0] = 0  # 虚拟边界
    dp[1] = nums[0]

    for i in range(2, n + 1):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])

    return dp[n]
```

### 3.3 负无穷初始化

```python
def with_negative_inf(arr):
    """求最大值时，初始化为负无穷"""
    n = len(arr)
    dp = [float('-inf')] * n
    dp[0] = arr[0]

    for i in range(1, n):
        dp[i] = max(arr[i], dp[i-1] + arr[i])

    return max(dp)  # 注意：可能全为负数
```

### 3.4 正无穷初始化

```python
def with_positive_inf(weights, values, W):
    """求最小值时，初始化为正无穷"""
    dp = [float('inf')] * (W + 1)
    dp[0] = 0  # 关键边界

    for i in range(len(weights)):
        for j in range(weights[i], W + 1):
            dp[j] = min(dp[j], dp[j - weights[i]] + values[i])

    return dp[W] if dp[W] != float('inf') else -1
```

## 4. 复杂度分析

边界初始化本身是O(1)到O(n)的操作，不影响整体复杂度。

## 5. 典型例题中的边界

### 5.1 编辑距离

```python
def editDistance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 边界：空串变换
    for i in range(m + 1): dp[i][0] = i  # 删除所有字符
    for j in range(n + 1): dp[0][j] = j  # 插入所有字符

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

### 5.2 正则表达式匹配

```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 关键边界：空串和含*的模式
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    # ... 递推
```

## 6. 常见陷阱与调试

### 6.1 常见陷阱

1. **空数组未处理**：n=0时访问dp[0]越界
2. **单元素未处理**：n=1时循环不执行，dp未正确初始化
3. **dp[0]含义不清**：dp[0]代表空集还是第一个元素？
4. **初始化值错误**：求最大值初始化为0（当答案可能为负时错误）
5. **恰好装满**：需要初始化为正无穷/负无穷

### 6.2 调试方法

1. **打印dp数组**：查看边界值是否正确
2. **手算小数据**：n=0, 1, 2, 3手动计算验证
3. **检查dp[0]**：确保初始条件符合题意
4. **检查转移边界**：i-1, i-2等是否会导致越界

### 6.3 初始化检查清单

```
□ dp数组大小是否正确？
□ dp[0] / dp[0][0] 是否初始化？
□ 特殊情况是否处理（n=0, n=1）？
□ 初始化值是否正确（0/inf/-inf/1）？
□ 边界循环是否正确处理？
□ 转移时是否可能越界？
```
