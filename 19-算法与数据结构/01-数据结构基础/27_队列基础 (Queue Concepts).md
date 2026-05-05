## Queue Concepts


```javascript
队列是一种先进先出（FIFO）的线性数据结构，新元素在队尾添加，从队头移除。```


```
class Queue {
  constructor() { this.items = []; }
  enqueue(x) { this.items.push(x); }
  dequeue() { return this.items.shift(); }
  front() { return this.items[0]; }
  isEmpty() { return this.items.length === 0; }
  size() { return this.items.length; }
}
const q = new Queue();
q.enqueue(1); q.enqueue(2); q.enqueue(3);
console.log(q.dequeue()); // 1
console.log(q.front());   // 2```


  点击按钮查看结果
