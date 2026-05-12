# 30-优先级队列 (Priority Queue)

优先级队列中每个元素都有优先级，高优先级元素先出队。通常用堆（Heap）实现。

## 最小堆实现

```javascript
class MinHeap {
  constructor() { this.heap = []; }

  size() { return this.heap.length; }
  peek() { return this.heap[0]; }

  push(val) {
    this.heap.push(val);
    this._bubbleUp(this.heap.length - 1);
  }

  pop() {
    if (!this.heap.length) return null;
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length) {
      this.heap[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  _bubbleUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.heap[parent] <= this.heap[i]) break;
      [this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
      i = parent;
    }
  }

  _sinkDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.heap[l] < this.heap[smallest]) smallest = l;
      if (r < n && this.heap[r] < this.heap[smallest]) smallest = r;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
}
```

## C++ 实现

```cpp
#include <queue>
#include <vector>
using namespace std;

// STL 优先级队列
priority_queue<int> maxPQ;              // 最大堆（默认）
priority_queue<int, vector<int>, greater<int>> minPQ; // 最小堆

int main() {
    minPQ.push(3); minPQ.push(1); minPQ.push(4);
    minPQ.push(1); minPQ.push(5);
    // 出队顺序: 1, 1, 3, 4, 5

    // 自定义比较器
    // 按第二个元素排序
    using PII = pair<int, int>;
    priority_queue<PII, vector<PII>, greater<PII>> pq;
    pq.push({2, 10}); pq.push({1, 20});
    auto [priority, value] = pq.top(); // {1, 20}
}
```

## 带比较器的优先队列

```javascript
class PriorityQueue {
  constructor(comparator = (a, b) => a - b) {
    this.heap = [];
    this.comparator = comparator;
  }

  push(val) {
    this.heap.push(val);
    this._bubbleUp(this.heap.length - 1);
  }

  pop() {
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length) {
      this.heap[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  peek() { return this.heap[0]; }
  size() { return this.heap.length; }

  _bubbleUp(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.comparator(this.heap[p], this.heap[i]) <= 0) break;
      [this.heap[p], this.heap[i]] = [this.heap[i], this.heap[p]];
      i = p;
    }
  }

  _sinkDown(i) {
    const n = this.heap.length;
    while (true) {
      let s = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.comparator(this.heap[l], this.heap[s]) < 0) s = l;
      if (r < n && this.comparator(this.heap[r], this.heap[s]) < 0) s = r;
      if (s === i) break;
      [this.heap[i], this.heap[s]] = [this.heap[s], this.heap[i]];
      i = s;
    }
  }
}
```

## 时间复杂度

| 操作 | 时间 | 说明 |
|------|------|------|
| push | O(log n) | 上浮调整 |
| pop | O(log n) | 下沉调整 |
| peek | O(1) | 查看堆顶 |
| 建堆 | O(n) | Floyd 算法 |
| 排序 | O(n log n) | 堆排序 |

## 典型应用

```javascript
// 第 K 大元素
function findKthLargest(nums, k) {
  const minHeap = new MinHeap();
  for (const n of nums) {
    minHeap.push(n);
    if (minHeap.size() > k) minHeap.pop();
  }
  return minHeap.peek();
}

// 合并 K 个有序链表
function mergeKLists(lists) {
  const pq = new PriorityQueue((a, b) => a.val - b.val);
  for (const node of lists) if (node) pq.push(node);
  const dummy = { val: 0, next: null };
  let curr = dummy;
  while (pq.size()) {
    const node = pq.pop();
    curr.next = node;
    curr = curr.next;
    if (node.next) pq.push(node.next);
  }
  return dummy.next;
}
```

## 何时用优先队列

- 需要频繁获取最大/最小值
- Top-K 问题
- 合并有序序列
- Dijkstra / Prim 算法
- 任务调度（按优先级执行）

## 常见陷阱

1. **堆类型混淆**：最大堆 vs 最小堆
2. **比较器方向**：`(a, b) => a - b` 是最小堆
3. **空堆操作**：pop 前检查 size
4. **自定义对象**：比较器要正确处理对象属性
