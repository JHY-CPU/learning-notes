## Stack Basics


```javascript
栈是一种后进先出（LIFO）的线性数据结构，只允许在一端（栈顶）进行插入和删除操作。```


```
class Stack {
  constructor() { this.items = []; }
  push(x) { this.items.push(x); }
  pop() { return this.items.pop(); }
  peek() { return this.items[this.items.length-1]; }
  isEmpty() { return this.items.length === 0; }
  size() { return this.items.length; }
}
const s = new Stack();
s.push(1); s.push(2); s.push(3);
console.log(s.pop()); // 3
console.log(s.peek()); // 2```


  点击按钮查看结果
