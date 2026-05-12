# 12-快速排序优化 (Quick Sort Optimizations)

快速排序在面对特定数据时可能退化到 O(n²)。以下是几种经典优化策略。

## 优化1：三数取中法选基准

```javascript
function medianOfThree(arr, low, high) {
  const mid = (low + high) >> 1;
  if (arr[low] > arr[mid]) [arr[low], arr[mid]] = [arr[mid], arr[low]];
  if (arr[low] > arr[high]) [arr[low], arr[high]] = [arr[high], arr[low]];
  if (arr[mid] > arr[high]) [arr[mid], arr[high]] = [arr[high], arr[mid]];
  // 中值放到 high-1 位置作为 pivot
  [arr[mid], arr[high - 1]] = [arr[high - 1], arr[mid]];
  return arr[high - 1];
}
```

## 优化2：三路快排（处理大量重复元素）

```javascript
function quickSort3Way(arr, low = 0, high = arr.length - 1) {
  if (low >= high) return arr;
  const pivot = arr[low];
  let lt = low;      // arr[low..lt-1] < pivot
  let gt = high;     // arr[gt+1..high] > pivot
  let i = low + 1;   // arr[lt..i-1] == pivot
  while (i <= gt) {
    if (arr[i] < pivot) {
      [arr[lt], arr[i]] = [arr[i], arr[lt]];
      lt++; i++;
    } else if (arr[i] > pivot) {
      [arr[i], arr[gt]] = [arr[gt], arr[i]];
      gt--;
    } else {
      i++;
    }
  }
  quickSort3Way(arr, low, lt - 1);
  quickSort3Way(arr, gt + 1, high);
  return arr;
}
```

## 优化3：小数组切换插入排序

```javascript
const CUTOFF = 16;

function insertionSortRange(arr, low, high) {
  for (let i = low + 1; i <= high; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= low && arr[j] > key) { arr[j + 1] = arr[j]; j--; }
    arr[j + 1] = key;
  }
}

function optimizedQuickSort(arr, low = 0, high = arr.length - 1) {
  if (high - low < CUTOFF) {
    insertionSortRange(arr, low, high);
    return;
  }
  const pi = partition(arr, low, high);
  optimizedQuickSort(arr, low, pi - 1);
  optimizedQuickSort(arr, pi + 1, high);
}
```

## 优化4：尾递归优化（减少栈深度）

```javascript
function tailRecursiveQuickSort(arr, low = 0, high = arr.length - 1) {
  while (low < high) {
    const pi = partition(arr, low, high);
    if (pi - low < high - pi) {
      tailRecursiveQuickSort(arr, low, pi - 1);
      low = pi + 1;  // 尾递归优化为循环
    } else {
      tailRecursiveQuickSort(arr, pi + 1, high);
      high = pi - 1;
    }
  }
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

const int CUTOFF = 16;

void insertionSort(vector<int>& a, int l, int r) {
    for (int i = l + 1; i <= r; i++) {
        int key = a[i], j = i - 1;
        while (j >= l && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}

// 三路快排
void quickSort3Way(vector<int>& a, int l, int r) {
    if (r - l < CUTOFF) { insertionSort(a, l, r); return; }
    int lt = l, gt = r, i = l + 1;
    int pivot = a[l];
    while (i <= gt) {
        if (a[i] < pivot) swap(a[lt++], a[i++]);
        else if (a[i] > pivot) swap(a[i], a[gt--]);
        else i++;
    }
    quickSort3Way(a, l, lt - 1);
    quickSort3Way(a, gt + 1, r);
}
```

## 优化效果对比

| 优化 | 解决的问题 | 效果 |
|------|-----------|------|
| 三数取中 | 已排序数据退化 | 最坏概率大幅降低 |
| 三路快排 | 大量重复元素 | 重复元素 O(n) 处理 |
| 插入排序切换 | 小子数组 | 减少递归开销 |
| 尾递归 | 递归栈太深 | 栈深度 O(log n) |

## 常见陷阱

1. **三数取中仍不够**：极端数据仍可能退化，可用随机化
2. **三路快排边界**：lt/gt/i 的移动要仔细处理
3. **CUTOFF 选择**：通常 8-20 之间，取决于平台
