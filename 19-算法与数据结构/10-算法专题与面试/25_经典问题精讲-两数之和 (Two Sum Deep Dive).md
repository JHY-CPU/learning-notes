# 经典问题精讲-两数之和 (Two Sum Deep Dive)

## 一、问题演进

两数之和是面试第一题，但它有多个变种，形成一个完整的问题系列：

| 题目 | 难度 | 关键变化 |
|------|------|---------|
| 两数之和 (LC 1) | Easy | 无序数组，返回下标 |
| 两数之和II (LC 167) | Medium | 有序数组，返回下标 |
| 三数之和 (LC 15) | Medium | 三个数和为0 |
| 最接近的三数之和 (LC 16) | Medium | 最接近目标 |
| 四数之和 (LC 18) | Medium | 四个数和为目标 |
| K数之和 | Hard | 递归+剪枝 |

---

## 二、两数之和 (LeetCode 1)

### 2.1 暴力法 — O(n^2)

```python
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```

### 2.2 哈希表 — O(n)

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

### 2.3 排序+双指针 — O(n log n)

适用于只需要值不需要下标的情况。

```python
def two_sum_sorted(nums, target):
    nums_sorted = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums_sorted[left][1] + nums_sorted[right][1]
        if s == target:
            return [nums_sorted[left][0], nums_sorted[right][0]]
        elif s < target: left += 1
        else: right -= 1
```

---

## 三、三数之和 (LeetCode 15)

### 3.1 排序+双指针

```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue  # 去重
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
            elif total < 0: left += 1
            else: right -= 1
    return result
```

**复杂度：** 时间 $O(n^2)$，空间 $O(1)$（不计输出）。

### 3.2 去重技巧

```python
# i去重：if i > 0 and nums[i] == nums[i-1]: continue
# left去重：while left < right and nums[left] == nums[left+1]: left += 1
# right去重：while left < right and nums[right] == nums[right-1]: right -= 1
```

---

## 四、K数之和 (通用递归解法)

```python
def k_sum(nums, target, k):
    nums.sort()
    return k_sum_helper(nums, target, k, 0)

def k_sum_helper(nums, target, k, start):
    result = []
    if k == 2:
        # 两数之和双指针
        left, right = start, len(nums) - 1
        while left < right:
            total = nums[left] + nums[right]
            if total == target:
                result.append([nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
            elif total < target: left += 1
            else: right -= 1
    else:
        for i in range(start, len(nums) - k + 1):
            if i > start and nums[i] == nums[i-1]: continue
            # 剪枝
            if nums[i] * k > target: break
            if nums[-1] * k < target: break
            for subset in k_sum_helper(nums, target - nums[i], k - 1, i + 1):
                result.append([nums[i]] + subset)

    return result
```

---

## 五、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 两数之和 — 哈希表
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
        if (seen.count(target - nums[i]))
            return {seen[target - nums[i]], i};
        seen[nums[i]] = i;
    }
    return {};
}

// 三数之和
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    int n = nums.size();
    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue;
        int left = i+1, right = n-1;
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

---

## 六、复杂度总结

| 方法 | 时间 | 空间 |
|------|------|------|
| 两数之和(哈希) | $O(n)$ | $O(n)$ |
| 两数之和(排序) | $O(n \log n)$ | $O(1)$ |
| 三数之和 | $O(n^2)$ | $O(1)$ |
| 四数之和 | $O(n^3)$ | $O(1)$ |
| K数之和 | $O(n^{k-1})$ | $O(k)$ |
