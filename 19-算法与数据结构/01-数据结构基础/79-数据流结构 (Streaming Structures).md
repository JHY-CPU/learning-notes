## Streaming Structures


```javascript
数据流结构处理在线数据，支持实时查询，如流中位数、流频率等。```


```
// 流中位数（两个堆）
class MedianFinder {
  constructor() { this.maxHeap = []; this.minHeap = []; }
  _maxHeapify(arr, val) { /* push then bubble up for max heap */ }
  addNum(num) {
    if (!this.maxHeap.length || num <= this.maxHeap[0]) {
      this.maxHeap.push(num); this.maxHeap.sort((a,b)=>b-a);
    } else { this.minHeap.push(num); this.minHeap.sort((a,b)=>a-b); }
    if (this.maxHeap.length > this.minHeap.length + 1)
      this.minHeap.push(this.maxHeap.shift());
    else if (this.minHeap.length > this.maxHeap.length)
      this.maxHeap.push(this.minHeap.shift());
  }
  findMedian() {
    if (this.maxHeap.length > this.minHeap.length) return this.maxHeap[0];
    return (this.maxHeap[0] + this.minHeap[0]) / 2;
  }
}```


  点击按钮查看结果
