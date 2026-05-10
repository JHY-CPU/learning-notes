# 经典问题精讲-最大子数组 (Maximum Subarray Deep Dive)

## 一、问题系列

| 题目 | 难度 | 变化 |
|------|------|------|
| 最大子数组和 (LC 53) | Medium | 基础版本 |
| 乘积最大子数组 (LC 152) | Medium | 乘法而非加法 |
| 最大子数组和II | — | 返回子数组本身 |
| 环形子数组最大和 (LC 918) | Medium | 环形数组 |
| 和最大的K个子数组 | — | TopK变种 |

---

## 二、最大子数组和 (LeetCode 53)

### 2.1 Kadane算法 — O(n)

**核心思想：** 以每个位置结尾的最大子数组和。

```python
def max_subarray(nums):
    max_sum = cur_sum = nums[0]
    for x in nums[1:]:
        cur_sum = max(x, cur_sum + x)  # 续接 or 重新开始
        max_sum = max(max_sum, cur_sum)
    return max_sum
```

**理解：** `cur_sum` 表示"以当前元素结尾的最大子数组和"。如果之前的累加和是负数，不如从当前元素重新开始。

### 2.2 同时记录区间

```python
def max_subarray_with_range(nums):
    max_sum = cur_sum = nums[0]
    start = end = 0
    temp_start = 0

    for i in range(1, len(nums)):
        if cur_sum + nums[i] < nums[i]:
            cur_sum = nums[i]
            temp_start = i
        else:
            cur_sum += nums[i]

        if cur_sum > max_sum:
            max_sum = cur_sum
            start = temp_start
            end = i

    return max_sum, nums[start:end+1]
```

### 2.3 分治法 — O(n log n)

```python
def max_subarray_dc(nums):
    def solve(left, right):
        if left == right:
            return nums[left]

        mid = (left + right) // 2

        # 左半最大
        left_max = solve(left, mid)
        # 右半最大
        right_max = solve(mid + 1, right)

        # 跨中点的最大
        left_cross = float('-inf')
        s = 0
        for i in range(mid, left - 1, -1):
            s += nums[i]
            left_cross = max(left_cross, s)

        right_cross = float('-inf')
        s = 0
        for i in range(mid + 1, right + 1):
            s += nums[i]
            right_cross = max(right_cross, s)

        return max(left_max, right_max, left_cross + right_cross)

    return solve(0, len(nums) - 1)
```

---

## 三、乘积最大子数组 (LeetCode 152)

**关键区别：** 负数乘负数得正数，所以需要同时维护最大值和最小值。

```python
def max_product(nums):
    max_prod = min_prod = result = nums[0]

    for x in nums[1:]:
        if x < 0:
            max_prod, min_prod = min_prod, max_prod

        max_prod = max(x, max_prod * x)
        min_prod = min(x, min_prod * x)
        result = max(result, max_prod)

    return result
```

**为什么需要维护最小值？** 因为最小的负数乘以一个负数可能变成最大的正数。

---

## 四、环形子数组最大和 (LeetCode 918)

**两种情况：**
1. 最大子数组在中间（普通 Kadane）
2. 最大子数组跨越首尾（总和 - 最小子数组和）

```python
def max_subarray_circular(nums):
    total = sum(nums)
    max_sum = cur_max = nums[0]
    min_sum = cur_min = nums[0]

    for x in nums[1:]:
        cur_max = max(x, cur_max + x)
        max_sum = max(max_sum, cur_max)
        cur_min = min(x, cur_min + x)
        min_sum = min(min_sum, cur_min)

    if max_sum < 0:
        return max_sum  # 全是负数

    return max(max_sum, total - min_sum)
```

---

## 五、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// Kadane算法
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], curSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        curSum = max(nums[i], curSum + nums[i]);
        maxSum = max(maxSum, curSum);
    }
    return maxSum;
}

// 乘积最大子数组
int maxProduct(vector<int>& nums) {
    int maxP = nums[0], minP = nums[0], result = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] < 0) swap(maxP, minP);
        maxP = max(nums[i], maxP * nums[i]);
        minP = min(nums[i], minP * nums[i]);
        result = max(result, maxP);
    }
    return result;
}
```

---

## 六、复杂度分析

| 算法 | 时间 | 空间 |
|------|------|------|
| Kadane | $O(n)$ | $O(1)$ |
| 分治 | $O(n \log n)$ | $O(\log n)$ |
| 乘积最大子数组 | $O(n)$ | $O(1)$ |
| 环形最大子数组 | $O(n)$ | $O(1)$ |

---

## 七、面试要点

1. **能否想到Kadane算法** — 最关键
2. **理解 cur_sum 的含义** — "以当前位置结尾的最大和"
3. **乘积变种** — 正负号处理
4. **环形变种** — 转化为"总和 - 最小子数组和"
