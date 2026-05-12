# 08-双指针技术 (Two Pointers Technique)

双指针技术使用两个指针遍历数组（通常同向或相向移动），将 O(n^2) 的暴力解法优化到 O(n)。

## 三种双指针模式

```javascript
// 模式1：同向双指针（快慢指针）
// 两指针从同一端出发，速度不同
function removeDuplicates(nums) {
  if (nums.length === 0) return 0;
  let slow = 0;
  for (let fast = 1; fast < nums.length; fast++) {
    if (nums[fast] !== nums[slow]) {
      slow++;
      nums[slow] = nums[fast];
    }
  }
  return slow + 1;
}

// 模式2：相向双指针（对撞指针）
// 两指针从两端向中间移动
function twoSumSorted(nums, target) {
  let left = 0, right = nums.length - 1;
  while (left < right) {
    let sum = nums[left] + nums[right];
    if (sum === target) return [left, right];
    if (sum < target) left++;
    else right--;
  }
  return [-1, -1];
}

// 模式3：分离双指针
// 两个指针在不同数组上独立移动
function mergeSortedArrays(A, B) {
  let i = 0, j = 0, result = [];
  while (i < A.length && j < B.length) {
    if (A[i] <= B[j]) result.push(A[i++]);
    else result.push(B[j++]);
  }
  while (i < A.length) result.push(A[i++]);
  while (j < B.length) result.push(B[j++]);
  return result;
}
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

// 移除有序数组重复元素
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < nums.size(); fast++) {
        if (nums[fast] != nums[slow]) {
            nums[++slow] = nums[fast];
        }
    }
    return slow + 1;
}

// 有序数组两数之和
vector<int> twoSum(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return {l, r};
        if (sum < target) l++;
        else r--;
    }
    return {};
}

// 移动零
void moveZeroes(vector<int>& nums) {
    int slow = 0;
    for (int fast = 0; fast < nums.size(); fast++) {
        if (nums[fast] != 0) {
            swap(nums[slow++], nums[fast]);
        }
    }
}
```

## 经典应用

```javascript
// 移动零：将数组中所有0移到末尾，保持非零元素相对顺序
function moveZeroes(nums) {
  let slow = 0;
  for (let fast = 0; fast < nums.length; fast++) {
    if (nums[fast] !== 0) {
      [nums[slow], nums[fast]] = [nums[fast], nums[slow]];
      slow++;
    }
  }
}

// 盛最多水的容器
function maxArea(height) {
  let l = 0, r = height.length - 1, max = 0;
  while (l < r) {
    max = Math.max(max, Math.min(height[l], height[r]) * (r - l));
    if (height[l] < height[r]) l++;
    else r--;
  }
  return max;
}

// 接雨水
function trap(height) {
  let l = 0, r = height.length - 1;
  let lMax = 0, rMax = 0, water = 0;
  while (l < r) {
    lMax = Math.max(lMax, height[l]);
    rMax = Math.max(rMax, height[r]);
    if (lMax < rMax) water += lMax - height[l++];
    else water += rMax - height[r--];
  }
  return water;
}
```

## 何时使用双指针

- 有序数组的搜索问题
- 原地修改数组（去重、移动元素）
- 子数组/子序列问题
- 回文判断
- 链表中环的检测

## 常见陷阱

1. **有序前提**：对撞指针通常要求数组有序
2. **边界条件**：注意 `left < right` 还是 `left <= right`
3. **去重处理**：三数之和等题目中要跳过重复元素
4. **指针移动顺序**：先移动再判断 vs 先判断再移动
