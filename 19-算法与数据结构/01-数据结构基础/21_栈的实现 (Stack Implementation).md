# Stack Implementation

### 什么是栈的实现方式

栈可以用数组或链表作为底层数据结构来实现，各有优劣。数组实现利用连续内存，缓存友好；链表实现天然支持动态扩容。

### 关键特性

- **数组实现**：随机访问快，但需要预分配空间或动态扩容
- **链表实现**：天然动态扩容，无需预估容量，但有额外指针开销
- **JavaScript Map 实现**：利用 Map 的有序性，性能介于两者之间

### 时间与空间复杂度

| 实现方式 | push | pop | peek | 空间开销 |
|---------|------|-----|------|---------|
| 数组 | O(1) 均摊 | O(1) | O(1) | 连续内存，可能浪费 |
| 链表 | O(1) | O(1) | O(1) | 每节点额外指针空间 |

### 适用场景

- **数组实现**：元素数量可预估，追求缓存性能
- **链表实现**：元素数量不确定，频繁扩容缩容
- **受限环境**：链表实现不需要大块连续内存

### 常见陷阱

- 数组扩容时的性能抖动（均摊 O(1) 但单次可能 O(n)）
- 链表实现忘记更新指针导致内存泄漏
- JavaScript 中数组的 shift/unshift 是 O(n)，不应作为队列底层

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
}
```


### 实际应用

- **系统级栈**：操作系统通常用链表实现调用栈，因为栈帧大小不固定
- **脚本语言**：V8 等引擎用数组实现执行栈，利用 CPU 缓存加速
- **面试常考**：用两个栈实现队列、用队列实现栈等经典问题
- **内存受限场景**：用链表实现避免数组预分配浪费

  点击按钮查看结果
