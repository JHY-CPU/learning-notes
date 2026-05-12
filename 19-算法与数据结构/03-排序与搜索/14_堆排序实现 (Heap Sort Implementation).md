# 15-堆排序实现 (Heap Sort Implementation)

堆排序分为两个阶段：建堆和排序。建堆后将堆顶与末尾交换，调整剩余元素为堆，重复此过程。

## JavaScript 实现

```javascript
// 迭代版 siftDown（避免递归栈）
function siftDown(arr, n, i) {
  while (true) {
    let largest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < n && arr[left] > arr[largest]) largest = left;
    if (right < n && arr[right] > arr[largest]) largest = right;
    if (largest === i) break;
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    i = largest;
  }
}

function heapSort(arr) {
  const n = arr.length;
  // 建最大堆
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    siftDown(arr, n, i);
  }
  // 排序
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]];
    siftDown(arr, i, 0);
  }
  return arr;
}

// 最小堆实现（用于求前K大）
class MinHeap {
  constructor(data = []) {
    this.data = data;
    for (let i = Math.floor(data.length / 2) - 1; i >= 0; i--) this._down(i);
  }
  peek() { return this.data[0]; }
  push(x) { this.data.push(x); this._up(this.data.length - 1); }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) { this.data[0] = last; this._down(0); }
    return top;
  }
  _up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p] <= this.data[i]) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  _down(i) {
    const n = this.data.length;
    while (true) {
      let s = i, l = 2*i+1, r = 2*i+2;
      if (l < n && this.data[l] < this.data[s]) s = l;
      if (r < n && this.data[r] < this.data[s]) s = r;
      if (s === i) break;
      [this.data[s], this.data[i]] = [this.data[i], this.data[s]];
      i = s;
    }
  }
}

// 测试
console.log(heapSort([12, 11, 13, 5, 6, 7])); // [5, 6, 7, 11, 12, 13]

const heap = new MinHeap([5, 3, 8, 1, 2]);
console.log(heap.pop()); // 1
console.log(heap.pop()); // 2
console.log(heap.pop()); // 3
```

## C++ 实现

```cpp
#include <vector>
#include <queue>
using namespace std;

// STL 优先队列
void demo() {
    // 最大堆
    priority_queue<int> maxHeap;
    maxHeap.push(5); maxHeap.push(3); maxHeap.push(8);
    // maxHeap.top() = 8

    // 最小堆
    priority_queue<int, vector<int>, greater<int>> minHeap;
    minHeap.push(5); minHeap.push(3); minHeap.push(8);
    // minHeap.top() = 3
}
```

## 堆排序 vs 其他排序

| 特性 | 堆排序 | 快排 | 归并 |
|------|--------|------|------|
| 最坏时间 | O(n log n) | O(n²) | O(n log n) |
| 空间 | O(1) | O(log n) | O(n) |
| 稳定 | 否 | 否 | 是 |
| 缓存友好 | 差 | 好 | 中 |

## 常见陷阱

1. **siftDown 终止条件**：largest === i 时停止
2. **排序阶段范围**：siftDown(arr, i, 0) 中 i 逐渐缩小
3. **适用场景**：适合需要 O(1) 空间的排序，不适合需要稳定排序的场景
