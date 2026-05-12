# 79-数据流结构 (Streaming Structures)

数据流结构处理在线数据，支持实时查询，如流中位数、流频率、流 Top-K 等。

## 流中位数（两个堆）

```javascript
class MedianFinder {
  constructor() {
    this.maxHeap = []; // 左半部分（最大堆）
    this.minHeap = []; // 右半部分（最小堆）
  }

  // 最大堆辅助函数
  _maxPush(val) {
    this.maxHeap.push(val);
    let i = this.maxHeap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.maxHeap[p] >= this.maxHeap[i]) break;
      [this.maxHeap[p], this.maxHeap[i]] = [this.maxHeap[i], this.maxHeap[p]];
      i = p;
    }
  }

  _maxPop() {
    const top = this.maxHeap[0];
    const last = this.maxHeap.pop();
    if (this.maxHeap.length) {
      this.maxHeap[0] = last;
      let i = 0;
      while (true) {
        let largest = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < this.maxHeap.length && this.maxHeap[l] > this.maxHeap[largest]) largest = l;
        if (r < this.maxHeap.length && this.maxHeap[r] > this.maxHeap[largest]) largest = r;
        if (largest === i) break;
        [this.maxHeap[i], this.maxHeap[largest]] = [this.maxHeap[largest], this.maxHeap[i]];
        i = largest;
      }
    }
    return top;
  }

  // 最小堆辅助函数（类似，略）

  addNum(num) {
    if (!this.maxHeap.length || num <= this.maxHeap[0]) {
      this._maxPush(num);
    } else {
      // minHeap push
      this.minHeap.push(num);
      this.minHeap.sort((a, b) => a - b); // 简化
    }
    // 平衡两个堆
    if (this.maxHeap.length > this.minHeap.length + 1) {
      this.minHeap.push(this._maxPop());
      this.minHeap.sort((a, b) => a - b);
    } else if (this.minHeap.length > this.maxHeap.length) {
      this._maxPush(this.minHeap.shift());
    }
  }

  findMedian() {
    if (this.maxHeap.length > this.minHeap.length) return this.maxHeap[0];
    return (this.maxHeap[0] + this.minHeap[0]) / 2;
  }
}
```

## 数据流中的第 K 大元素

```javascript
class KthLargest {
  constructor(k, nums) {
    this.k = k;
    this.heap = []; // 最小堆
    for (const n of nums) this.add(n);
  }

  add(val) {
    this._push(val);
    if (this.heap.length > this.k) this._pop();
    return this.heap[0];
  }

  _push(val) {
    this.heap.push(val);
    let i = this.heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.heap[p] <= this.heap[i]) break;
      [this.heap[p], this.heap[i]] = [this.heap[i], this.heap[p]];
      i = p;
    }
  }

  _pop() {
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length) {
      this.heap[0] = last;
      let i = 0;
      while (true) {
        let s = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < this.heap.length && this.heap[l] < this.heap[s]) s = l;
        if (r < this.heap.length && this.heap[r] < this.heap[s]) s = r;
        if (s === i) break;
        [this.heap[i], this.heap[s]] = [this.heap[s], this.heap[i]];
        i = s;
      }
    }
    return top;
  }
}
```

## 数据流频率统计

```javascript
// Top-K 高频元素
class TopKFrequent {
  constructor() {
    this.freq = new Map();
  }

  record(item) {
    this.freq.set(item, (this.freq.get(item) || 0) + 1);
  }

  topK(k) {
    // 用最小堆维护 Top-K
    const heap = [];
    for (const [item, count] of this.freq) {
      heap.push([count, item]);
      heap.sort((a, b) => a[0] - b[0]);
      if (heap.length > k) heap.shift();
    }
    return heap.reverse().map(([count, item]) => item);
  }
}
```

## 数据流滑动窗口

```javascript
class SlidingWindowStats {
  constructor(windowSize) {
    this.windowSize = windowSize;
    this.queue = []; // {val, time}
    this.sum = 0;
  }

  add(val, timestamp) {
    this.queue.push({ val, timestamp });
    this.sum += val;
    // 移除过期元素
    const cutoff = timestamp - this.windowSize;
    while (this.queue.length && this.queue[0].timestamp <= cutoff) {
      this.sum -= this.queue.shift().val;
    }
  }

  average() {
    return this.queue.length ? this.sum / this.queue.length : 0;
  }

  count() { return this.queue.length; }
  sumValue() { return this.sum; }
}
```

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 流中位数 add | O(log n) | O(n) |
| 流中位数 find | O(1) | - |
| 第K大 add | O(log k) | O(k) |
| 频率统计 record | O(1) | O(n) |
| 滑动窗口 add | O(1) 均摊 | O(w) |

## 应用场景

- 实时监控系统（CPU、内存、网络指标）
- 搜索引擎热词统计
- 股票实时价格分析
- 日志实时分析
- 在线推荐系统
- 网络流量监控

## 常见陷阱

1. **内存增长**：数据流是无限的，必须有淘汰策略
2. **时间窗口**：要注意时间戳的单位和精度
3. **并发安全**：多线程写入需要加锁
4. **精度问题**：浮点数累加可能有误差
