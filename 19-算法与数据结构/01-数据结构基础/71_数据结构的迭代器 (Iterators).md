# Iterators

### 什么是迭代器

迭代器是一种设计模式，提供统一的接口遍历不同数据结构的元素，而无需暴露底层实现。遵循"迭代器协议"，通常包含 hasNext() 和 next() 方法。

### 关键特性

- **解耦遍历**：使用者不关心数据结构是数组、树还是图
- **惰性求值**：按需生成下一个元素，节省内存
- **可组合**：多个迭代器可以串联、过滤、映射
- **JavaScript 原生支持**：可迭代对象实现 Symbol.iterator 方法

### 时间与空间复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 创建迭代器 | O(1) | O(h) h 为树高 |
| next() | O(1) 均摊 | - |
| 完整遍历 | O(n) | O(h) |

### 适用场景 vs 替代方案

- **树的遍历**：迭代器避免递归栈溢出
- **大数据集**：惰性迭代避免一次性加载全部数据
- **流式处理**：逐条处理记录而非批量加载
- **替代**：小数据量直接用 for...of 更简洁

### 常见陷阱

- 迭代过程中修改集合会导致未定义行为
- 忘记检查 hasNext() 直接调用 next() 可能报错
- 递归迭代器的栈空间消耗可能很大

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
}
```


### 实际应用

- **Java Iterator**：集合框架的标准遍历接口
- **Python 生成器**：yield 关键字实现惰性迭代
- **数据库游标**：逐行读取结果集而非全部加载
- **文件处理**：逐行读取大文件避免内存溢出

  点击按钮查看结果
