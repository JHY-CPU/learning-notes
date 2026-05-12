# 16-二分查找基础 (Binary Search)

二分查找在有序数组中 O(log n) 时间定位目标。

## 标准模板

```javascript
function binarySearch(arr, target) {
  let l = 0, r = arr.length - 1;
  while (l <= r) {
    const mid = l + ((r - l) >> 1); // 防溢出
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int binarySearch(vector<int>& arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1;
}
```

## 二分查找变体

```javascript
// 1. 查找第一个等于 target 的位置
function lowerBound(arr, target) {
  let l = 0, r = arr.length;
  while (l < r) {
    const mid = l + ((r - l) >> 1);
    if (arr[mid] < target) l = mid + 1;
    else r = mid;
  }
  return l;
}

// 2. 查找最后一个等于 target 的位置
function upperBound(arr, target) {
  let l = 0, r = arr.length;
  while (l < r) {
    const mid = l + ((r - l) >> 1);
    if (arr[mid] <= target) l = mid + 1;
    else r = mid;
  }
  return l - 1;
}

// 3. 查找插入位置（第一个 >= target 的位置）
function searchInsert(arr, target) {
  let l = 0, r = arr.length;
  while (l < r) {
    const mid = l + ((r - l) >> 1);
    if (arr[mid] < target) l = mid + 1;
    else r = mid;
  }
  return l;
}

// 4. 旋转排序数组搜索
function searchRotated(nums, target) {
  let l = 0, r = nums.length - 1;
  while (l <= r) {
    const mid = l + ((r - l) >> 1);
    if (nums[mid] === target) return mid;
    if (nums[l] <= nums[mid]) {
      if (target >= nums[l] && target < nums[mid]) r = mid - 1;
      else l = mid + 1;
    } else {
      if (target > nums[mid] && target <= nums[r]) l = mid + 1;
      else r = mid - 1;
    }
  }
  return -1;
}
```

## 二分答案

```javascript
// 问题：最小化最大值
// 例：分割数组的最大值最小化
function splitArray(nums, m) {
  let l = Math.max(...nums), r = nums.reduce((a, b) => a + b, 0);
  while (l < r) {
    const mid = l + ((r - l) >> 1);
    if (canSplit(nums, m, mid)) r = mid;
    else l = mid + 1;
  }
  return l;
}

function canSplit(nums, m, maxSum) {
  let count = 1, sum = 0;
  for (const n of nums) {
    if (sum + n > maxSum) { count++; sum = 0; }
    sum += n;
  }
  return count <= m;
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 标准二分 | O(log n) | O(1) |
| 递归二分 | O(log n) | O(log n) |
| 二分答案 | O(n log V) | O(1) |

## 常见陷阱

1. **死循环**：`l = mid` 在 `l + 1 == r` 时会死循环，应用 `l = mid + 1`
2. **溢出**：`(l + r) / 2` 可能溢出，用 `l + (r - l) / 2`
3. **边界选择**：`l <= r` 与 `l < r` 的使用场景不同
4. **闭区间 vs 开区间**：统一用一种风格避免混乱
5. **数组为空**：要提前处理空数组
