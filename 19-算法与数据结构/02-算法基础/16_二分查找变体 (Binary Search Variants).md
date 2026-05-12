# 17-二分查找变体 (Binary Search Variants)

标准二分查找只能找到任意一个目标位置，变体写法可解决更多问题：找第一个、找最后一个、找插入位置等。

## 常见变体

| 变体 | 含义 | 关键调整 |
|------|------|----------|
| 找左边界 | 第一个等于 target 的位置 | 找到后继续往左缩 |
| 找右边界 | 最后一个等于 target 的位置 | 找到后继续往右缩 |
| 找插入位置 | 第一个 >= target 的位置 | Lower Bound |
| 找第一个 > target | Upper Bound | 严格大于 |
| 旋转数组最小值 | 找旋转排序数组的最小值 | 比较 mid 与 right |

## Lower Bound 与 Upper Bound

Lower Bound 返回第一个 >= target 的位置，Upper Bound 返回第一个 > target 的位置。它们是所有变体的基础模板。

## JavaScript 实现

```javascript
// Lower Bound：第一个 >= target 的位置
function lowerBound(arr, target) {
  let l = 0, r = arr.length;
  while (l < r) {
    const mid = l + Math.floor((r - l) / 2);
    if (arr[mid] < target) l = mid + 1;
    else r = mid;
  }
  return l;
}

// Upper Bound：第一个 > target 的位置
function upperBound(arr, target) {
  let l = 0, r = arr.length;
  while (l < r) {
    const mid = l + Math.floor((r - l) / 2);
    if (arr[mid] <= target) l = mid + 1;
    else r = mid;
  }
  return l;
}

// 查找第一个等于 target 的位置
function firstEqual(arr, target) {
  const idx = lowerBound(arr, target);
  if (idx < arr.length && arr[idx] === target) return idx;
  return -1;
}

// 查找最后一个等于 target 的位置
function lastEqual(arr, target) {
  const idx = upperBound(arr, target) - 1;
  if (idx >= 0 && arr[idx] === target) return idx;
  return -1;
}

// 统计 target 出现次数
function countEqual(arr, target) {
  return upperBound(arr, target) - lowerBound(arr, target);
}

// 查找旋转数组最小值
function findMinRotated(arr) {
  let l = 0, r = arr.length - 1;
  while (l < r) {
    const mid = l + Math.floor((r - l) / 2);
    if (arr[mid] > arr[r]) l = mid + 1;
    else r = mid;
  }
  return arr[l];
}

// 测试
console.log(lowerBound([1, 2, 2, 2, 3, 4], 2));  // 1
console.log(upperBound([1, 2, 2, 2, 3, 4], 2));  // 4
console.log(firstEqual([1, 2, 2, 2, 3, 4], 2));  // 1
console.log(lastEqual([1, 2, 2, 2, 3, 4], 2));   // 3
console.log(countEqual([1, 2, 2, 2, 3, 4], 2));  // 3
console.log(findMinRotated([4, 5, 6, 7, 0, 1, 2])); // 0
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// 使用 STL
// lower_bound: 第一个 >= target 的迭代器
// upper_bound: 第一个 > target 的迭代器

int firstEqual(vector<int>& arr, int target) {
    auto it = lower_bound(arr.begin(), arr.end(), target);
    if (it != arr.end() && *it == target) return it - arr.begin();
    return -1;
}

int lastEqual(vector<int>& arr, int target) {
    auto it = upper_bound(arr.begin(), arr.end(), target);
    if (it != arr.begin() && *(it - 1) == target) return (it - 1) - arr.begin();
    return -1;
}

// 手写 lower_bound
int myLowerBound(vector<int>& arr, int target) {
    int l = 0, r = arr.size();
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] < target) l = mid + 1;
        else r = mid;
    }
    return l;
}

// 旋转数组找最小值
int findMinRotated(vector<int>& arr) {
    int l = 0, r = arr.size() - 1;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] > arr[r]) l = mid + 1;
        else r = mid;
    }
    return arr[l];
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| Lower/Upper Bound | O(log n) | O(1) |
| 查找边界 | O(log n) | O(1) |
| 统计次数 | O(log n) | O(1) |
| 旋转数组最小值 | O(log n) | O(1) |

## 易错点

1. **循环条件**：`l < r` vs `l <= r`，前者适合开区间 [l, r)，后者适合闭区间 [l, r]
2. **mid 计算**：用 `l + (r - l) / 2` 防止溢出
3. **边界更新**：lower_bound 用 `r = mid`（mid 可能是答案），普通二分用 `r = mid - 1`
4. **空数组处理**：先判断数组是否为空

## 实际应用

二分变体是面试高频考点。JavaScript 的 `Array.prototype.findIndex` 等价于 Lower Bound 思想。数据库索引查找、有序集合的插入位置都使用这些变体。STL 的 `lower_bound` / `upper_bound` 在竞赛中频繁使用。
