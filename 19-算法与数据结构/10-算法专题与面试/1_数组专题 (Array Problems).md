# 数组专题 (Array Problems)

## 一、概念定义与原理

### 1.1 数组基础

数组是最基础的数据结构，内存中连续存储，支持 $O(1)$ 随机访问。

**核心操作的时间复杂度：**
- 访问：$O(1)$
- 查找：$O(n)$（无序）、$O(\log n)$（有序+二分）
- 插入/删除：$O(n)$（需要移动元素）

### 1.2 数组类问题的核心技巧

| 技巧 | 适用场景 | 时间复杂度 |
|------|---------|-----------|
| 双指针 | 有序数组、去重、两数之和 | $O(n)$ |
| 滑动窗口 | 连续子数组满足条件 | $O(n)$ |
| 前缀和 | 区间和查询 | $O(n)$ 预处理，$O(1)$ 查询 |
| 差分数组 | 区间增减操作 | $O(n)$ 预处理，$O(1)$ 修改 |
| 排序预处理 | 需要有序性 | $O(n \log n)$ |

---

## 二、核心算法详解

### 2.1 双指针技术

**对撞指针：** 从两端向中间移动
```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]
```

**快慢指针：** 同向移动，速度不同
```python
def remove_duplicates(nums):
    if not nums: return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

### 2.2 滑动窗口

**模板：**
```python
def sliding_window(s, condition):
    window = {}  # 窗口内的状态
    left = 0
    result = 0

    for right in range(len(s)):
        # 扩大窗口
        c = s[right]
        window[c] = window.get(c, 0) + 1

        # 收缩窗口（不满足条件时）
        while not valid(window, condition):
            d = s[left]
            window[d] -= 1
            left += 1

        # 更新结果
        result = max(result, right - left + 1)

    return result
```

### 2.3 前缀和

**一维前缀和：**
```python
class PrefixSum:
    def __init__(self, nums):
        n = len(nums)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i+1] = self.prefix[i] + nums[i]

    def range_sum(self, left, right):
        """返回 nums[left..right] 的和"""
        return self.prefix[right+1] - self.prefix[left]
```

**二维前缀和：**
```python
class PrefixSum2D:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                self.prefix[i+1][j+1] = (matrix[i][j]
                    + self.prefix[i][j+1]
                    + self.prefix[i+1][j]
                    - self.prefix[i][j])

    def region_sum(self, r1, c1, r2, c2):
        return (self.prefix[r2+1][c2+1]
              - self.prefix[r1][c2+1]
              - self.prefix[r2+1][c1]
              + self.prefix[r1][c1])
```

### 2.4 差分数组

```python
class DifferenceArray:
    def __init__(self, nums):
        n = len(nums)
        self.diff = [0] * n
        self.diff[0] = nums[0]
        for i in range(1, n):
            self.diff[i] = nums[i] - nums[i-1]

    def increment(self, left, right, val):
        """对 nums[left..right] 每个元素加 val"""
        self.diff[left] += val
        if right + 1 < len(self.diff):
            self.diff[right+1] -= val

    def result(self):
        nums = [0] * len(self.diff)
        nums[0] = self.diff[0]
        for i in range(1, len(self.diff)):
            nums[i] = nums[i-1] + self.diff[i]
        return nums
```

---

## 三、经典题目详解

### 3.1 合并区间 (LeetCode 56)

**思路：** 按起始位置排序，逐个合并重叠区间。

```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    for (auto& inv : intervals) {
        if (merged.empty() || merged.back()[1] < inv[0])
            merged.push_back(inv);
        else
            merged.back()[1] = max(merged.back()[1], inv[1]);
    }
    return merged;
}
```

### 3.2 轮转数组 (LeetCode 189)

**三次翻转法：** 先整体翻转，再分别翻转两部分。

```python
def rotate(nums, k):
    n = len(nums)
    k %= n
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])
```

### 3.3 接雨水 (LeetCode 42)

**双指针法：** 每个位置的水量 = min(左侧最高, 右侧最高) - 当前高度。

```cpp
int trap(vector<int>& height) {
    int n = height.size(), water = 0;
    int left = 0, right = n - 1;
    int lmax = 0, rmax = 0;
    while (left < right) {
        lmax = max(lmax, height[left]);
        rmax = max(rmax, height[right]);
        if (lmax < rmax)
            water += lmax - height[left++];
        else
            water += rmax - height[right--];
    }
    return water;
}
```

### 3.4 缺失的第一个正数 (LeetCode 41)

**原地哈希：** 将数组本身当作哈希表，值为 `i` 的数放到下标 `i-1` 的位置。

```python
def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:
            nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1
```

---

## 四、代码实现汇总

### 4.1 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 前缀和查询
class NumArray {
    vector<int> prefix;
public:
    NumArray(vector<int>& nums) {
        prefix.resize(nums.size() + 1, 0);
        for (int i = 0; i < nums.size(); i++)
            prefix[i+1] = prefix[i] + nums[i];
    }
    int sumRange(int left, int right) {
        return prefix[right+1] - prefix[left];
    }
};

// 三数之和
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    int n = nums.size();
    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int left = i + 1, right = n - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.push_back({nums[i], nums[left], nums[right]});
                while (left < right && nums[left] == nums[left+1]) left++;
                while (left < right && nums[right] == nums[right-1]) right--;
                left++; right--;
            } else if (sum < 0) left++;
            else right--;
        }
    }
    return result;
}
```

### 4.2 Python 实现

```python
# 最大子数组和 (Kadane算法)
def max_subarray(nums):
    max_sum = cur_sum = nums[0]
    for x in nums[1:]:
        cur_sum = max(x, cur_sum + x)
        max_sum = max(max_sum, cur_sum)
    return max_sum

# 乘积最大子数组
def max_product(nums):
    max_prod = min_prod = result = nums[0]
    for x in nums[1:]:
        if x < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(x, max_prod * x)
        min_prod = min(x, min_prod * x)
        result = max(result, max_prod)
    return result

# 下一个排列
def next_permutation(nums):
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i+1:] = reversed(nums[i+1:])
```

---

## 五、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 双指针 | $O(n)$ | $O(1)$ |
| 滑动窗口 | $O(n)$ | $O(k)$ k为字符集大小 |
| 前缀和 | $O(n)$ 预处理 | $O(n)$ |
| 差分数组 | $O(1)$ 单次修改 | $O(n)$ |
| Kadane算法 | $O(n)$ | $O(1)$ |
| 合并区间 | $O(n \log n)$ | $O(n)$ |
| 接雨水(双指针) | $O(n)$ | $O(1)$ |

---

## 六、面试高频题

1. **LeetCode 1：** 两数之和
2. **LeetCode 15：** 三数之和
3. **LeetCode 56：** 合并区间
4. **LeetCode 42：** 接雨水
5. **LeetCode 53：** 最大子数组和
6. **LeetCode 189：** 轮转数组
7. **LeetCode 41：** 缺失的第一个正数
8. **LeetCode 238：** 除自身以外数组的乘积
9. **LeetCode 48：** 旋转图像
10. **LeetCode 54：** 螺旋矩阵
