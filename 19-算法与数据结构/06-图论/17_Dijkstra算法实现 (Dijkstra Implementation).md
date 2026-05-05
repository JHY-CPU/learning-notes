## Dijkstra 算法（优先队列优化）

  使用最小堆（优先队列）优化，每次 O(logV) 取出最小距离顶点，总复杂度 O((V+E)logV)。


```javascript
class MinHeap {
  constructor() { this.heap = []; }
  push(v, dist) {
    this.heap.push({v, dist});
    this.bubbleUp(this.heap.length-1);
  }
  pop() {
    if (this.heap.length===1) return this.heap.pop();
    const top = this.heap[0];
    this.heap[0] = this.heap.pop();
    this.bubbleDown(0);
    return top;
  }
  bubbleUp(i) {
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.heap[i].dist >= this.heap[parent].dist) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }
  bubbleDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.heap[l].dist < this.heap[smallest].dist) smallest = l;
      if (r < n && this.heap[r].dist < this.heap[smallest].dist) smallest = r;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
  get size() { return this.heap.length; }
}```

  ## 交互演示
