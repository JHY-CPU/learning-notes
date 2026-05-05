## Iterators


```javascript
迭代器是一种设计模式，提供统一的方式遍历不同的数据结构。```


```
class ArrayIterator {
  constructor(arr) { this.arr = arr; this.idx = 0; }
  hasNext() { return this.idx < this.arr.length; }
  next() { return this.hasNext() ? this.arr[this.idx++] : null; }
}
class TreeNode {
  constructor(val) { this.val = val; this.left = null; this.right = null; }
}
// 二叉树中序遍历迭代器
class InorderIterator {
  constructor(root) { this.stack = []; this._pushLeft(root); }
  _pushLeft(node) { while (node) { this.stack.push(node); node = node.left; } }
  hasNext() { return this.stack.length > 0; }
  next() { const node = this.stack.pop(); this._pushLeft(node.right); return node.val; }
}```


  点击按钮查看结果
