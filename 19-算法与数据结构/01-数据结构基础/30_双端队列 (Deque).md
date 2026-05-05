## Deque


```javascript
双端队列允许在两端进行插入和删除操作，结合了栈和队列的功能。```


```
class Deque {
  constructor() { this.data = {}; this.front = 0; this.back = 0; }
  addFront(x) { this.data[--this.front] = x; }
  addBack(x) { this.data[this.back++] = x; }
  removeFront() { if (this.isEmpty()) return null; const v = this.data[this.front]; delete this.data[this.front++]; return v; }
  removeBack() { if (this.isEmpty()) return null; const v = this.data[--this.back]; delete this.data[this.back]; return v; }
  peekFront() { return this.data[this.front]; }
  peekBack() { return this.data[this.back-1]; }
  isEmpty() { return this.front === this.back; }
  size() { return this.back - this.front; }
}```


  点击按钮查看结果
