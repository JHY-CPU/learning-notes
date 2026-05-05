## 17-单链表实现 (Singly Linked List Implementation)

单链表的完整实现，包括增删改查、按索引操作等核心方法。

## 完整单链表实现

```javascript

class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

class SinglyLinkedList {
  constructor() {
    // 虚拟头节点简化边界处理
    this.dummyHead = new ListNode(null);
    this.size = 0;
  }

  // 获取索引为 index 的节点（0-based）
  getNode(index) {
    this._checkIndex(index);
    let curr = this.dummyHead.next;
    for (let i = 0; i < index; i++) {
      curr = curr.next;
    }
    return curr;
  }

  // 在指定索引插入
  addAtIndex(index, val) {
    if (index < 0 || index > this.size) return false;
    let prev = this.dummyHead;
    for (let i = 0; i < index; i++) {
      prev = prev.next;
    }
    let newNode = new ListNode(val);
    newNode.next = prev.next;
    prev.next = newNode;
    this.size++;
    return true;
  }

  // 删除指定索引的节点
  removeAtIndex(index) {
    this._checkIndex(index);
    let prev = this.dummyHead;
    for (let i = 0; i < index; i++) {
      prev = prev.next;
    }
    let removed = prev.next;
    prev.next = removed.next;
    removed.next = null; // 辅助 GC
    this.size--;
    return removed.val;
  }

  // 头部插入 / 尾部插入快捷方法
  addFirst(val) { return this.addAtIndex(0, val); }
  addLast(val) { return this.addAtIndex(this.size, val); }

  // 遍历链表
  toArray() {
    let result = [];
    let curr = this.dummyHead.next;
    while (curr) {
      result.push(curr.val);
      curr = curr.next;
    }
    return result;
  }

  // 内部：检查索引有效性
  _checkIndex(index) {
    if (index < 0 || index >= this.size) {
      throw new Error(`索引越界: ${index}, 大小: ${this.size}`);
    }
  }
}
```

## 交互演示
