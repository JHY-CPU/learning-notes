## Stack Implementation


```javascript
栈可以使用数组或链表实现。数组实现简单高效，链表实现动态扩容灵活。```


```
// 链表实现栈
class Node {
  constructor(val) { this.val = val; this.next = null; }
}
class LinkedStack {
  constructor() { this.top = null; this._size = 0; }
  push(x) {
    const n = new Node(x);
    n.next = this.top;
    this.top = n;
    this._size++;
  }
  pop() {
    if (!this.top) return null;
    const v = this.top.val;
    this.top = this.top.next;
    this._size--;
    return v;
  }
  peek() { return this.top ? this.top.val : null; }
  size() { return this._size; }
}```


  点击按钮查看结果
