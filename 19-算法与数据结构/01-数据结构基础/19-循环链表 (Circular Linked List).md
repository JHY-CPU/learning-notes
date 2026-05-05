## 19-循环链表 (Circular Linked List)

循环链表是链表的变体，尾节点的 next 指向头节点，形成环状结构。可以是单向或双向。

## 循环链表特点

```javascript

// 循环链表的核心特征：
// 1. 尾节点的 next 指向头节点（而非 null）
// 2. 从任意节点出发都能遍历整个链表
// 3. 适合需要反复循环遍历的场景
// 4. 约瑟夫问题、轮转调度等经典应用

class CircularListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

class CircularLinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
    this.size = 0;
  }

  // 尾部插入
  addLast(val) {
    let newNode = new CircularListNode(val);
    if (!this.head) {
      // 第一个节点，指向自身
      this.head = newNode;
      this.tail = newNode;
      newNode.next = newNode; // 自环
    } else {
      newNode.next = this.head;
      this.tail.next = newNode;
      this.tail = newNode;
    }
    this.size++;
  }

  // 删除指定值的节点
  remove(val) {
    if (!this.head) return false;

    let curr = this.head;
    let prev = this.tail;

    do {
      if (curr.val === val) {
        if (this.size === 1) {
          this.head = null;
          this.tail = null;
        } else {
          prev.next = curr.next;
          if (curr === this.head) this.head = curr.next;
          if (curr === this.tail) this.tail = prev;
        }
        this.size--;
        return true;
      }
      prev = curr;
      curr = curr.next;
    } while (curr !== this.head);

    return false;
  }

  // 遍历（限制步数防止无限循环）
  traverse(steps) {
    if (!this.head) return [];
    let result = [];
    let curr = this.head;
    for (let i = 0; i < steps; i++) {
      result.push(curr.val);
      curr = curr.next;
    }
    return result;
  }
}
```

## 交互演示
