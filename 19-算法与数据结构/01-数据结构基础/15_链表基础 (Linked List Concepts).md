# 16-链表基础 (Linked List Concepts)

链表是一种线性数据结构，元素通过指针链接而非连续内存存储。每个节点包含数据和指向下一个节点的指针。

## 链表 vs 数组

```
数组：连续内存，随机访问 O(1)，插入删除 O(n)
链表：分散内存，顺序访问 O(n)，插入删除 O(1)（已找到位置）
```

## 链表节点定义

```javascript
class ListNode {
  constructor(val, next = null) {
    this.val = val;
    this.next = next;
  }
}
```

## C++ 节点定义

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
```

## 链表类实现

```javascript
class LinkedList {
  constructor() {
    this.head = null;
    this.size = 0;
  }

  // 头部插入 O(1)
  addFirst(val) {
    this.head = new ListNode(val, this.head);
    this.size++;
  }

  // 尾部插入 O(n)（无尾指针时）
  addLast(val) {
    let newNode = new ListNode(val);
    if (!this.head) {
      this.head = newNode;
    } else {
      let curr = this.head;
      while (curr.next) curr = curr.next;
      curr.next = newNode;
    }
    this.size++;
  }

  // 按索引插入 O(n)
  addAt(index, val) {
    if (index < 0 || index > this.size) return;
    if (index === 0) return this.addFirst(val);
    let prev = this.head;
    for (let i = 0; i < index - 1; i++) prev = prev.next;
    prev.next = new ListNode(val, prev.next);
    this.size++;
  }

  // 删除头节点 O(1)
  removeFirst() {
    if (!this.head) return;
    this.head = this.head.next;
    this.size--;
  }

  // 打印
  print() {
    let curr = this.head, result = [];
    while (curr) { result.push(curr.val); curr = curr.next; }
    return result.join(' -> ') + ' -> null';
  }
}
```

## 时间复杂度对比

| 操作 | 数组 | 链表 |
|------|------|------|
| 按索引访问 | O(1) | O(n) |
| 按值查找 | O(n) | O(n) |
| 头部插入 | O(n) | O(1) |
| 尾部插入 | O(1) 均摊 | O(n)* |
| 中间插入 | O(n) | O(1)** |
| 空间开销 | 较小 | 每节点有指针开销 |

*有尾指针时 O(1)
**已找到插入位置

## 链表的优缺点

**优点**：
- 动态大小，无需预分配
- 插入删除不需要移动元素
- 可以高效实现栈、队列

**缺点**：
- 不支持随机访问
- 额外指针空间开销
- 缓存不友好（非连续内存）

## 常见陷阱

1. **空链表处理**：操作前检查 head 是否为 null
2. **丢失引用**：修改指针前要保存后续节点引用
3. **循环链表**：遍历时需要有终止条件
4. **内存泄漏**：删除节点后要断开引用

## 何时选择链表

- 频繁在头部或中间插入删除
- 不需要随机访问
- 数据量不确定
- 实现其他数据结构（栈、队列、图的邻接表）
