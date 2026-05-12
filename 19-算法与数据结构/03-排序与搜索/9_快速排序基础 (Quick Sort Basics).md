# 10-快速排序基础 (Quick Sort Basics)

快速排序由 Tony Hoare 于 1959 年提出，是目前应用最广泛的排序算法之一。它采用分治策略，通过分区操作将数组分为两部分递归排序。

## 核心思想

选择一个基准元素（pivot），将数组重新排列：小于 pivot 的放左边，大于 pivot 的放右边，然后递归排序左右两部分。

## 算法步骤

1. **选择基准**：从数组中选择一个元素作为 pivot
2. **分区（Partition）**：重新排列数组，所有小于 pivot 的元素放左边，大于的放右边
3. **递归排序**：对左右两个子数组递归应用快速排序

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 最好 | O(n log n) | O(log n) |
| 平均 | O(n log n) | O(log n) |
| 最坏（已排序） | O(n²) | O(n) |

稳定性：不稳定。原地排序：是。

## JavaScript 实现

```javascript
// Lomuto 分区方案（选最后一个元素为基准）
function quickSort(arr, low = 0, high = arr.length - 1) {
  if (low < high) {
    const pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
  return arr;
}

function partition(arr, low, high) {
  const pivot = arr[high];
  let i = low - 1;
  for (let j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
  return i + 1;
}

// Hoare 分区方案（效率更高）
function quickSortHoare(arr, low = 0, high = arr.length - 1) {
  if (low < high) {
    const pi = partitionHoare(arr, low, high);
    quickSortHoare(arr, low, pi);
    quickSortHoare(arr, pi + 1, high);
  }
  return arr;
}

function partitionHoare(arr, low, high) {
  const pivot = arr[Math.floor((low + high) / 2)];
  let i = low - 1, j = high + 1;
  while (true) {
    do { i++; } while (arr[i] < pivot);
    do { j--; } while (arr[j] > pivot);
    if (i >= j) return j;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

console.log(quickSort([10, 7, 8, 9, 1, 5]));       // [1, 5, 7, 8, 9, 10]
console.log(quickSortHoare([10, 7, 8, 9, 1, 5]));   // [1, 5, 7, 8, 9, 10]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

## Lomuto vs Hoare

| 特性 | Lomuto | Hoare |
|------|--------|-------|
| 基准位置 | 最后元素 | 中间元素 |
| 交换次数 | 较多 | 较少（约3倍少） |
| 实现难度 | 简单 | 稍复杂 |
| 返回值 | pivot 最终位置 | 分割点 |

## 常见陷阱

1. **最坏情况**：已排序数组 + 固定选最后一个元素 = O(n²)
2. **分区边界**：Hoare 返回的 j 不是 pivot 最终位置
3. **递归栈**：最坏递归深度 O(n)，可能栈溢出
