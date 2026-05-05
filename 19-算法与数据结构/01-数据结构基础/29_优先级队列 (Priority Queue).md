## Priority Queue


```javascript
优先级队列中每个元素都有优先级，高优先级元素先出队。通常用堆实现。```


```
// 最小堆实现优先级队列
class MinPQ {
  constructor() { this.heap = []; }
  push(x) {
    this.heap.push(x);
    let i = this.heap.length - 1;
    while (i > 0) {
      const p = Math.floor((i-1)/2);
      if (this.heap[p] <= this.heap[i]) break;
      [this.heap[p], this.heap[i]] = [this.heap[i], this.heap[p]];
      i = p;
    }
  }
  pop() {
    if (!this.heap.length) return null;
    const r = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length) {
      this.heap[0] = last;
      let i = 0;
      while (true) {
        let smallest = i;
        const l = 2*i+1, r2 = 2*i+2;
        if (l < this.heap.length && this.heap[l] < this.heap[smallest]) smallest = l;
        if (r2 < this.heap.length && this.heap[r2] < this.heap[smallest]) smallest = r2;
        if (smallest === i) break;
        [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
        i = smallest;
      }
    }
    return r;
  }
}```


  点击按钮查看结果
