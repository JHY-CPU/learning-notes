# 戳气球 (Burst Balloons)

## 1. 概念与定义

戳气球问题（LeetCode 312）：给定 n 个气球，每个气球上有一个数字。戳破第 i 个气球可以得到 `nums[i-1] * nums[i] * nums[i+1]` 的硬币。求戳破所有气球能获得的最大硬币数。

关键洞察：**正向思考（先戳哪个）很困难，逆向思考（最后戳哪个）更自然。** 这是一个经典的区间DP问题。

## 2. 状态定义与转移方程

### 2.1 逆向思考

```
dp[i][j] = 戳破开区间 (i, j) 内所有气球的最大硬币数（不包括i和j）
假设 k 是最后戳破的气球：
  dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
  for k in (i+1, ..., j-1)

初始条件：dp[i][i+1] = 0（相邻气球之间没有气球）
答案：dp[0][n+1]（nums首尾添加1作为边界）
```

### 2.2 为什么逆向思考更简单

正向思考时，戳破一个气球会改变相邻气球的乘积因子，导致子问题之间互相影响。

逆向思考时，假设 k 是最后被戳破的气球，那么 (i, k) 和 (k, j) 是两个独立的子问题。

## 3. 算法实现

### 3.1 标准区间DP

```python
def maxCoins(nums):
    nums = [1] + nums + [1]  # 添加虚拟边界
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):  # 区间长度
        for left in range(n - length):
            right = left + length
            for k in range(left + 1, right):
                coins = nums[left] * nums[k] * nums[right]
                dp[left][right] = max(dp[left][right],
                                      dp[left][k] + dp[k][right] + coins)

    return dp[0][n - 1]
```

### 3.2 记忆化搜索

```python
from functools import lru_cache

def maxCoins_memo(nums):
    nums = [1] + nums + [1]
    n = len(nums)

    @lru_cache(maxsize=None)
    def dp(left, right):
        """戳破 (left, right) 内所有气球的最大收益"""
        if left + 1 >= right:
            return 0
        result = 0
        for k in range(left + 1, right):
            coins = nums[left] * nums[k] * nums[right]
            result = max(result, dp(left, k) + dp(k, right) + coins)
        return result

    return dp(0, n - 1)
```

### 3.3 C++ 实现

```cpp
int maxCoins(vector<int>& nums) {
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    int n = nums.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len < n; len++)
        for (int left = 0; left + len < n; left++) {
            int right = left + len;
            for (int k = left + 1; k < right; k++)
                dp[left][right] = max(dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left]*nums[k]*nums[right]);
        }

    return dp[0][n-1];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 区间DP | O(n³) | O(n²) |
| 记忆化搜索 | O(n³) | O(n²) |

## 5. 典型例题

### 例题1：移除盒子（LeetCode 546）

```python
def removeBoxes(boxes):
    """移除连续相同颜色的盒子，得分 = k²"""
    n = len(boxes)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(l, r, k):
        """boxes[l..r]，且boxes[r]右边有k个与boxes[r]同色的盒子"""
        if l > r:
            return 0
        # 合并右侧同色盒子
        while r > l and boxes[r] == boxes[r - 1]:
            r -= 1
            k += 1
        # 选择1：直接移除 boxes[r] 及其右侧k个
        result = dp(l, r - 1, 0) + (k + 1) ** 2
        # 选择2：在 [l, r-1] 中找与 boxes[r] 同色的位置 i，合并
        for i in range(l, r):
            if boxes[i] == boxes[r]:
                result = max(result, dp(l, i, k + 1) + dp(i + 1, r - 1, 0))

        return result

    return dp(0, n - 1, 0)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **正向思考陷阱**：正向DP会导致子问题相互影响
2. **边界处理**：首尾添加1作为虚拟边界，简化转移
3. **区间长度**：从2开始（相邻元素之间至少长度为2）

### 6.2 区间DP的关键

- **逆向思考**：考虑最后一步
- **独立子问题**：分割后子问题互不影响
- **枚举顺序**：按区间长度从小到大

### 6.3 相关问题

- 戳气球（LeetCode 312）
- 移除盒子（LeetCode 546）
- 多边形三角剖分（LeetCode 1039）
- 石子合并（经典问题）
