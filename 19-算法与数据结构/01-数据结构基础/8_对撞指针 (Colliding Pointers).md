# 09-对撞指针 (Colliding Pointers)

对撞指针（相向双指针）是指两个指针从数组两端向中间移动，常用于有序数组的搜索和反转问题。

## 对撞指针模式

```javascript
function collidingPointers(arr, target) {
  let left = 0;
  let right = arr.length - 1;
  while (left < right) {
    let current = compute(left, right);
    if (current === target) return [left, right];
    else if (current < target) left++;
    else right--;
  }
  return [-1, -1];
}
```

## C++ 实现

```cpp
#include <vector>
#include <string>
using namespace std;

// 两数之和 II（有序数组）
vector<int> twoSum(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return {l + 1, r + 1};
        if (sum < target) l++;
        else r--;
    }
    return {};
}

// 验证回文
bool isPalindrome(string s) {
    int l = 0, r = s.size() - 1;
    while (l < r) {
        while (l < r && !isalnum(s[l])) l++;
        while (l < r && !isalnum(s[r])) r--;
        if (tolower(s[l++]) != tolower(s[r--])) return false;
    }
    return true;
}

// 盛最多水的容器
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1, maxA = 0;
    while (l < r) {
        maxA = max(maxA, min(height[l], height[r]) * (r - l));
        if (height[l] < height[r]) l++;
        else r--;
    }
    return maxA;
}
```

## 经典例题

```javascript
// 1. 两数之和 II（有序数组）
function twoSum(numbers, target) {
  let l = 0, r = numbers.length - 1;
  while (l < r) {
    let sum = numbers[l] + numbers[r];
    if (sum === target) return [l + 1, r + 1];
    if (sum < target) l++;
    else r--;
  }
  return [-1, -1];
}

// 2. 验证回文串
function isPalindrome(s) {
  s = s.toLowerCase().replace(/[^a-z0-9]/g, '');
  let l = 0, r = s.length - 1;
  while (l < r) {
    if (s[l] !== s[r]) return false;
    l++; r--;
  }
  return true;
}

// 3. 盛最多水的容器
function maxArea(height) {
  let l = 0, r = height.length - 1, max = 0;
  while (l < r) {
    let area = Math.min(height[l], height[r]) * (r - l);
    max = Math.max(max, area);
    if (height[l] < height[r]) l++;
    else r--;
  }
  return max;
}

// 4. 有序数组的平方
function sortedSquares(nums) {
  let result = new Array(nums.length);
  let l = 0, r = nums.length - 1, idx = r;
  while (l <= r) {
    if (Math.abs(nums[l]) > Math.abs(nums[r])) {
      result[idx--] = nums[l] * nums[l++];
    } else {
      result[idx--] = nums[r] * nums[r--];
    }
  }
  return result;
}
```

## 对撞指针 vs 暴力解法

| 问题 | 暴力 | 对撞指针 |
|------|------|----------|
| 两数之和（有序） | O(n²) | O(n) |
| 盛水最多 | O(n²) | O(n) |
| 接雨水 | O(n²) | O(n) |
| 回文判断 | O(n) | O(n) 但常数更小 |

## 关键技巧

1. **移动较小边**：在盛水问题中，移动高度较小的指针才有可能找到更大面积
2. **跳过重复**：三数之和中，找到答案后跳过重复元素避免重复解
3. **从两端开始**：对撞指针要求数组可从两端访问，链表不适合
4. **提前终止**：某些问题可以在条件不满足时直接返回

## 常见陷阱

1. 忘记处理 `left < right` 边界条件
2. 移动指针后没有检查越界
3. 在需要去重的题目中没有跳过重复元素
4. 错误地判断移动哪边的指针
