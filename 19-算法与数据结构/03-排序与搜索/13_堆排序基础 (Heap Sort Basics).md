# 14-堆排序基础 (Heap Sort Basics)

堆排序利用堆这种数据结构进行排序。堆是一棵完全二叉树，可用数组高效存储。

## 堆的重要性质

- 父节点下标：`floor((i-1)/2)`
- 左子节点下标：`2*i + 1`
- 右子节点下标：`2*i + 2`
- 最后一个非叶子节点：`floor(n/2) - 1`

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 平均时间 | O(n log n) |
| 最坏时间 | O(n log n) |
| 最好时间 | O(n log n) |
| 建堆 | O(n) |
| 空间 | O(1) |
| 稳定性 | 不稳定 |

## JavaScript 实现

```javascript
// 下沉操作：维护最大堆性质
function siftDown(arr, n, i) {
  let largest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;
  if (left < n && arr[left] > arr[largest]) largest = left;
  if (right < n && arr[right] > arr[largest]) largest = right;
  if (largest !== i) {
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    siftDown(arr, n, largest);
  }
}

// 建堆：从最后一个非叶子节点开始下沉
function buildMaxHeap(arr) {
  const n = arr.length;
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    siftDown(arr, n, i);
  }
}

// 堆排序
function heapSort(arr) {
  const n = arr.length;
  // 1. 建最大堆 O(n)
  buildMaxHeap(arr);
  // 2. 逐个取出堆顶 O(n log n)
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]]; // 交换堆顶到末尾
    siftDown(arr, i, 0); // 调整剩余元素
  }
  return arr;
}

console.log(heapSort([12, 11, 13, 5, 6, 7])); // [5, 6, 7, 11, 12, 13]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void siftDown(vector<int>& a, int n, int i) {
    int largest = i, l = 2*i+1, r = 2*i+2;
    if (l < n && a[l] > a[largest]) largest = l;
    if (r < n && a[r] > a[largest]) largest = r;
    if (largest != i) {
        swap(a[i], a[largest]);
        siftDown(a, n, largest);
    }
}

void heapSort(vector<int>& a) {
    int n = a.size();
    for (int i = n/2 - 1; i >= 0; i--) siftDown(a, n, i);
    for (int i = n - 1; i > 0; i--) {
        swap(a[0], a[i]);
        siftDown(a, i, 0);
    }
}
```

## 建堆为什么是 O(n)

从最后一个非叶子节点开始下沉，每层节点数 n/2^(h+1)，下沉高度 h，总代价为 sum(h * n/2^(h+1)) = O(n)，不是 O(n log n)。

## 常见陷阱

1. **循环 vs 递归 siftDown**：递归版在极深堆时可能栈溢出
2. **数组索引**：注意 left = 2i+1 而不是 2i（0-indexed）
3. **稳定性**：堆排序不稳定
