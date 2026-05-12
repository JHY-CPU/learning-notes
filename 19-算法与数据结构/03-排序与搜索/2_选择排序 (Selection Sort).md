# 3-选择排序 (Selection Sort)

选择排序每次从未排序部分找到最小元素，将其放到已排序部分的末尾。

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 最好 | O(n²) | O(1) |
| 平均 | O(n²) | O(1) |
| 最坏 | O(n²) | O(1) |

稳定性：不稳定（例如 [5a, 5b, 3] 排序后 5b 可能跑到 5a 前面）。原地排序：是。

## JavaScript 实现

```javascript
// 基础选择排序
function selectionSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < n; j++) {
      if (arr[j] < arr[minIdx]) minIdx = j;
    }
    if (minIdx !== i) {
      [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
    }
  }
  return arr;
}

// 双向选择排序（同时找最大最小）
function doubleSelectionSort(arr) {
  const n = arr.length;
  for (let i = 0; i < Math.floor(n / 2); i++) {
    let minIdx = i, maxIdx = i;
    for (let j = i; j < n - i; j++) {
      if (arr[j] < arr[minIdx]) minIdx = j;
      if (arr[j] > arr[maxIdx]) maxIdx = j;
    }
    [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
    if (maxIdx === i) maxIdx = minIdx; // 修正最大值索引
    [arr[n - 1 - i], arr[maxIdx]] = [arr[maxIdx], arr[n - 1 - i]];
  }
  return arr;
}

console.log(selectionSort([64, 25, 12, 22, 11])); // [11, 12, 22, 25, 64]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void selectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIdx]) minIdx = j;
        }
        if (minIdx != i) swap(arr[i], arr[minIdx]);
    }
}
```

## 算法特点

- **交换次数少**：最多 n-1 次交换，优于冒泡排序的 O(n²) 次交换
- **比较次数固定**：总是 n(n-1)/2 次比较
- **不适应数据**：即使已排序也要做全部比较

## 适用场景

- 教学：原理简单直观
- 交换代价极高的场景：交换次数为 O(n)
- 小规模数据：元素较少时简单可用

## 常见陷阱

1. **稳定性误解**：选择排序不稳定
2. **与插入排序混淆**：两者都是 O(n²) 但插入排序对近乎有序数据更快
3. **适用性**：实际工程中几乎不使用选择排序
