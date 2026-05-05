## Circular Queue


```javascript
循环队列用数组和两个指针实现，避免数据搬移操作，空间利用率更高。```


```
class CircularQueue {
  constructor(k) {
    this.data = new Array(k);
    this.head = 0;
    this.tail = 0;
    this.cap = k;
  }
  enqueue(x) {
    if ((this.tail + 1) % this.cap === this.head) return false;
    this.data[this.tail] = x;
    this.tail = (this.tail + 1) % this.cap;
    return true;
  }
  dequeue() {
    if (this.head === this.tail) return false;
    this.head = (this.head + 1) % this.cap;
    return true;
  }
  front() { return this.head === this.tail ? -1 : this.data[this.head]; }
  isFull() { return (this.tail + 1) % this.cap === this.head; }
}```


  点击按钮查看结果
