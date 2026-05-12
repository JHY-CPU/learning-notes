# 13-快速选择 (Quick Select)

快速选择算法用于在未排序数组中查找第 K 大（或第 K 小）元素，基于快速排序的分区思想，只需递归处理一侧，平均 O(n)。

## 算法思路

1. 选择一个基准元素进行分区
2. 检查 pivot 位置：
   - 如果 pivotIndex == k，找到答案
   - 如果 pivotIndex > k，在左半部分继续
   - 如果 pivotIndex < k，在右半部分继续

## JavaScript 实现

```javascript
// 查找第 k 小元素 (0-indexed)
function quickSelect(arr, k) {
  function select(left, right) {
    if (left === right) return arr[left];
    const pi = partition(arr, left, right);
    if (k === pi) return arr[k];
    if (k < pi) return select(left, pi - 1);
    return select(pi + 1, right);
  }
  return select(0, arr.length - 1);
}

function partition(arr, low, high) {
  const pivot = arr[high];
  let i = low;
  for (let j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
    }
  }
  [arr[i], arr[high]] = [arr[high], arr[i]];
  return i;
}

// 第 K 大 = 第 (n - k) 小
function findKthLargest(arr, k) {
  return quickSelect(arr, arr.length - k);
}

// 迭代版（避免递归栈溢出）
function quickSelectIterative(arr, k) {
  let left = 0, right = arr.length - 1;
  while (left <= right) {
    const pi = partition(arr, left, right);
    if (pi === k) return arr[pi];
    if (pi < k) left = pi + 1;
    else right = pi - 1;
  }
  return arr[left];
}

// 随机化版本（避免最坏情况）
function quickSelectRandom(arr, k) {
  function select(left, right) {
    if (left === right) return arr[left];
    // 随机选 pivot
    const randIdx = left + Math.floor(Math.random() * (right - left + 1));
    [arr[randIdx], arr[right]] = [arr[right], arr[randIdx]];
    const pi = partition(arr, left, right);
    if (k === pi) return arr[k];
    if (k < pi) return select(left, pi - 1);
    return select(pi + 1, right);
  }
  return select(0, arr.length - 1);
}

// 测试
console.log(findKthLargest([3, 2, 1, 5, 6, 4], 2));  // 5
console.log(quickSelectIterative([3, 2, 1, 5, 6, 4], 4)); // 4 (第5小)
```

## C++ 实现

```cpp
#include <vector>
#include <cstdlib>
using namespace std;

int partition(vector<int>& a, int l, int r) {
    int pivot = a[r], i = l;
    for (int j = l; j < r; j++) {
        if (a[j] <= pivot) swap(a[i++], a[j]);
    }
    swap(a[i], a[r]);
    return i;
}

int quickSelect(vector<int>& a, int l, int r, int k) {
    if (l == r) return a[l];
    int pi = partition(a, l, r);
    if (k == pi) return a[k];
    if (k < pi) return quickSelect(a, l, pi - 1, k);
    return quickSelect(a, pi + 1, r, k);
}

int findKthLargest(vector<int>& a, int k) {
    return quickSelect(a, 0, a.size() - 1, a.size() - k);
}
```

## 复杂度

| 情况 | 时间 | 空间 |
|------|------|------|
| 平均 | O(n) | O(log n) |
| 最坏 | O(n²) | O(n) |
| 随机化期望 | O(n) | O(log n) |

## 与堆方法对比

| 方法 | 时间 | 空间 | 适用场景 |
|------|------|------|---------|
| 快速选择 | O(n) 平均 | O(1) | 找单个第K大 |
| 堆 | O(n log k) | O(k) | 找前K大（多个） |
| 排序 | O(n log n) | O(1) | 需要完整排序 |

## 常见陷阱

1. **k 的含义**：注意第 k 大 vs 第 k 小的区别
2. **最坏情况**：固定选最后一个元素 + 已排序 = O(n²)，随机化解决
3. **有副作用**：quickSelect 会部分修改数组顺序
