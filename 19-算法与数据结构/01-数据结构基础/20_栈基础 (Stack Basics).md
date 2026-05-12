# 21-栈基础 (Stack Basics)

栈（Stack）是一种后进先出（LIFO）的线性数据结构，只允许在栈顶进行插入（push）和删除（pop）操作。

## 核心特性

- **受限访问**：只能访问栈顶元素
- **LIFO 顺序**：最后压入的元素最先弹出
- **操作统一**：所有操作都在栈顶完成

## JavaScript 实现

```javascript
class Stack {
  constructor() { this.items = []; }
  push(x) { this.items.push(x); }
  pop() { return this.items.pop(); }
  peek() { return this.items[this.items.length - 1]; }
  isEmpty() { return this.items.length === 0; }
  size() { return this.items.length; }
  clear() { this.items = []; }
}

// 使用
const s = new Stack();
s.push(1); s.push(2); s.push(3);
console.log(s.pop());  // 3
console.log(s.peek()); // 2
console.log(s.size()); // 2
```

## C++ 实现

```cpp
#include <stack>
#include <iostream>
using namespace std;

int main() {
    stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    cout << s.top() << endl;  // 3
    s.pop();
    cout << s.top() << endl;  // 2
    cout << s.size() << endl; // 2
    cout << s.empty() << endl; // 0 (false)

    // 用数组实现栈
    // int stk[1000], top = -1;
    // stk[++top] = val;  // push
    // int val = stk[top--]; // pop
    // int val = stk[top];   // peek
}
```

## 基于链表的栈

```javascript
class LinkedStack {
  constructor() { this.top = null; this.sz = 0; }

  push(val) {
    this.top = { val, next: this.top };
    this.sz++;
  }

  pop() {
    if (!this.top) return undefined;
    let val = this.top.val;
    this.top = this.top.next;
    this.sz--;
    return val;
  }

  peek() { return this.top ? this.top.val : undefined; }
  isEmpty() { return this.top === null; }
  size() { return this.sz; }
}
```

## 时间复杂度

| 操作 | 时间 | 说明 |
|------|------|------|
| push | O(1) | 栈顶添加 |
| pop | O(1) | 栈顶移除 |
| peek/top | O(1) | 查看栈顶 |
| isEmpty | O(1) | 判空 |
| 空间 | O(n) | 存储 n 个元素 |

## 典型应用

```javascript
// 1. 括号匹配
function isValid(s) {
  let stack = [];
  let map = { ')': '(', ']': '[', '}': '{' };
  for (let c of s) {
    if (c in map) {
      if (stack.pop() !== map[c]) return false;
    } else {
      stack.push(c);
    }
  }
  return stack.length === 0;
}

// 2. 最小栈
class MinStack {
  constructor() {
    this.stack = [];
    this.minStack = [Infinity];
  }
  push(x) {
    this.stack.push(x);
    this.minStack.push(Math.min(x, this.minStack[this.minStack.length - 1]));
  }
  pop() {
    this.stack.pop();
    this.minStack.pop();
  }
  top() { return this.stack[this.stack.length - 1]; }
  getMin() { return this.minStack[this.minStack.length - 1]; }
}

// 3. 用两个栈实现队列
class QueueWithStacks {
  constructor() {
    this.inStack = [];
    this.outStack = [];
  }
  push(x) { this.inStack.push(x); }
  pop() {
    if (!this.outStack.length) {
      while (this.inStack.length) this.outStack.push(this.inStack.pop());
    }
    return this.outStack.pop();
  }
  peek() {
    if (!this.outStack.length) {
      while (this.inStack.length) this.outStack.push(this.inStack.pop());
    }
    return this.outStack[this.outStack.length - 1];
  }
}
```

## 适用场景

- 浏览器后退/前进
- 编辑器撤销/重做
- 深度优先搜索（DFS）
- 括号匹配、表达式求值
- 函数调用栈管理
- 单调栈求下一个更大元素

## 常见陷阱

1. **栈溢出**：递归过深导致栈空间耗尽
2. **空栈操作**：对空栈执行 pop 或 peek 会出错
3. **混淆 LIFO/FIFO**：栈是 LIFO，队列是 FIFO
4. **数组实现的扩容**：固定大小的栈需要考虑扩容
