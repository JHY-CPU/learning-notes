# 单调队列优化DP (Monotonic Queue DP)

## 1. 概念与定义

单调队列优化是利用**单调队列**数据结构，将某些DP问题的时间复杂度从 O(n²) 优化到 O(n) 的技巧。

单调队列维护一个单调递增或单调递减的序列。在DP转移中，当状态转移涉及"滑动窗口内的最值"时，可以用单调队列高效维护。

适用场景：
- 转移方程涉及**连续区间内的最大/最小值**
- 区间长度有上限（如窗口大小为 k）
- 典型问题：滑动窗口最大值、多重背包单调队列优化

## 2. 状态定义与转移方程

### 2.1 滑动窗口最大值

```
对于每个位置 i，求 [i-k+1, i] 范围内的最大值
用单调递减队列维护候选最大值
```

### 2.2 带限制的DP转移

```
dp[i] = min/max(dp[j]) + cost(i)  其中 j ∈ [i-k, i-1]
用单调队列维护 [i-k, i-1] 范围内的最优转移
```

### 2.3 多重背包

```
dp[j] = max(dp[j - k*w[i]] + k*v[i]) for k in [0, c[i]]
按模w[i]分组，每组用单调队列优化
```

## 3. 算法实现

### 3.1 单调队列模板

```python
from collections import deque

class MonotonicQueue:
    """单调递减队列（维护最大值）"""
    def __init__(self):
        self.q = deque()

    def push(self, val, idx):
        # 弹出所有小于val的元素
        while self.q and self.q[-1][0] <= val:
            self.q.pop()
        self.q.append((val, idx))

    def pop(self, idx, k):
        # 弹出超出窗口范围的元素
        while self.q and self.q[0][1] < idx - k + 1:
            self.q.popleft()

    def max(self):
        return self.q[0][0] if self.q else None
```

### 3.2 滑动窗口最大值（LeetCode 239）

```python
def maxSlidingWindow(nums, k):
    from collections import deque
    q = deque()  # 存储索引
    result = []

    for i, num in enumerate(nums):
        # 弹出所有小于当前值的元素
        while q and nums[q[-1]] <= num:
            q.pop()
        q.append(i)

        # 弹出超出窗口的元素
        if q[0] < i - k + 1:
            q.popleft()

        # 窗口形成后记录答案
        if i >= k - 1:
            result.append(nums[q[0]])

    return result
```

### 3.3 单调队列优化DP

```python
def constrainedDP(nums, k):
    """
    dp[i] = min(dp[j]) + nums[i]  其中 j ∈ [max(0,i-k), i-1]
    滑动窗口最小值的DP
    """
    from collections import deque
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    q = deque()
    q.append(0)

    for i in range(1, n):
        # 弹出超出窗口的元素
        while q and q[0] < i - k:
            q.popleft()
        # 转移
        dp[i] = dp[q[0]] + nums[i]
        # 维护单调递增队列（求最小值）
        while q and dp[q[-1]] >= dp[i]:
            q.pop()
        q.append(i)

    return dp[n - 1]
```

### 3.4 滑动窗口的子数组最大平均数（LeetCode 643）

```python
def findMaxAverage(nums, k):
    # 直接用滑动窗口，不需要单调队列
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum / k
```

### 3.5 C++ 实现

```cpp
// 滑动窗口最大值
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> q;  // 存索引
    vector<int> res;
    for (int i = 0; i < nums.size(); i++) {
        while (!q.empty() && nums[q.back()] <= nums[i]) q.pop_back();
        q.push_back(i);
        if (q.front() < i - k + 1) q.pop_front();
        if (i >= k - 1) res.push_back(nums[q.front()]);
    }
    return res;
}
```

## 4. 复杂度分析

| 问题 | 暴力复杂度 | 优化后复杂度 |
|------|-----------|-------------|
| 滑动窗口最大值 | O(nk) | O(n) |
| 带限制DP | O(nk) | O(n) |
| 多重背包 | O(nW*c) | O(nW) |

## 5. 典型例题

### 例题1：绝对差不超过限制的最长连续子数组（LeetCode 1438）

```python
def longestSubarray(nums, limit):
    from collections import deque
    max_q = deque()  # 单调递减
    min_q = deque()  # 单调递增
    left = 0
    result = 0

    for right, num in enumerate(nums):
        while max_q and nums[max_q[-1]] <= num:
            max_q.pop()
        max_q.append(right)

        while min_q and nums[min_q[-1]] >= num:
            min_q.pop()
        min_q.append(right)

        while nums[max_q[0]] - nums[min_q[0]] > limit:
            left += 1
            if max_q[0] < left:
                max_q.popleft()
            if min_q[0] < left:
                min_q.popleft()

        result = max(result, right - left + 1)

    return result
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **存索引而非值**：需要判断窗口范围，必须存索引
2. **队列为空的处理**：转移前检查队列非空
3. **单调性选择**：求最大值用递减队列，求最小值用递增队列
4. **窗口边界**：注意 left 和 right 的位置关系

### 6.2 单调队列与单调栈的区别

| 特性 | 单调队列 | 单调栈 |
|------|---------|--------|
| 用途 | 滑动窗口最值 | 下一个更大/更小元素 |
| 弹出条件 | 超出窗口范围 | 更小/更大的元素入栈 |
| 两端操作 | 两端都弹出 | 只从一端弹出 |

### 6.3 使用条件

- 转移方程包含 `min(dp[j])` 或 `max(dp[j])`
- j 的范围是连续区间
- 区间边界随 i 单调移动
