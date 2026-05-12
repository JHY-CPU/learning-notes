# Deque

### 什么是双端队列

双端队列（Deque, Double-Ended Queue）允许在队列的两端进行插入和删除操作，同时具备栈和队列的功能。是一种比普通队列更灵活的数据结构。

### 关键特性

- **四类操作**：addFront、addBack、removeFront、removeBack 均为 O(1)
- **通用性强**：可作为栈（只用一端）或队列（一端入一端出）使用
- **两端访问**：支持 peekFront 和 peekBack 快速查看

### 时间与空间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| addFront/addBack | O(1) | 两端插入 |
| removeFront/removeBack | O(1) | 两端删除 |
| peekFront/peekBack | O(1) | 两端查看 |
| 空间 | O(n) | 存储 n 个元素 |

### 适用场景 vs 替代方案

- **滑动窗口最值**：单调队列基于双端队列实现
- **回文检查**：从两端同时比较字符
- **撤销/重做**：支持在历史记录两端操作
- **替代**：只需要 LIFO 用栈，只需要 FIFO 用普通队列

### 常见陷阱

- JavaScript 没有内置 Deque，用数组模拟时注意 shift 的 O(n) 问题
- 用对象 + 双指针实现时，指针可能变为负数或很大，注意边界
- 并发环境下需要额外同步机制

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
}
```


### 实际应用

- **Python collections.deque**：标准库中高效的双端队列实现
- **Java ArrayDeque**：替代 Stack 类的推荐选择
- **浏览器历史**：前进/后退功能可用双端队列管理
- **滑动窗口算法**：单调队列的核心底层结构

  点击按钮查看结果
