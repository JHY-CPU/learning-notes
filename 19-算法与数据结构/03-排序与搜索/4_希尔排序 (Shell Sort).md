# 5-希尔排序 (Shell Sort)

希尔排序是插入排序的改进版，通过将数组按增量分组，对每组进行插入排序，逐步缩小增量直到 1。

## 复杂度分析

复杂度取决于增量序列：

| 增量序列 | 最坏时间 | 说明 |
|---------|---------|------|
| Shell 原始 (n/2, n/4...) | O(n²) | 最差 |
| Knuth (1, 4, 13, 40...) | O(n^1.5) | 实用 |
| Sedgewick | O(n^(4/3)) | 较好 |
| 最优序列 | O(n log²n) | 理论最优 |

空间 O(1)，不稳定排序。

## JavaScript 实现

```javascript
// Shell 原始增量序列
function shellSort(arr) {
  const n = arr.length;
  for (let gap = Math.floor(n / 2); gap > 0; gap = Math.floor(gap / 2)) {
    for (let i = gap; i < n; i++) {
      const temp = arr[i];
      let j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
  return arr;
}

// Knuth 增量序列
function shellSortKnuth(arr) {
  const n = arr.length;
  let gap = 1;
  while (gap < n / 3) gap = gap * 3 + 1; // 1, 4, 13, 40, 121...

  while (gap >= 1) {
    for (let i = gap; i < n; i++) {
      const temp = arr[i];
      let j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
    gap = Math.floor(gap / 3);
  }
  return arr;
}

console.log(shellSort([12, 34, 54, 2, 3]));       // [2, 3, 12, 34, 54]
console.log(shellSortKnuth([12, 34, 54, 2, 3]));   // [2, 3, 12, 34, 54]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void shellSort(vector<int>& arr) {
    int n = arr.size();
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j = i;
            while (j >= gap && arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}
```

## 算法原理

希尔排序的核心是"预排序"：通过大步长让元素快速移动到大致正确的位置，然后逐步缩小步长做精细调整。当 gap=1 时退化为插入排序，但此时数组已基本有序，插入排序接近 O(n)。

## 适用场景

- 中等规模数据：比插入排序快，实现简单
- 内存受限：O(1) 空间
- 不需要稳定性：希尔排序不稳定

## 常见陷阱

1. **增量选择**：不当的增量序列会导致性能退化
2. **稳定性**：希尔排序不稳定
3. **与插入排序比较**：gap=1 时就是插入排序，但此时数据已基本有序
