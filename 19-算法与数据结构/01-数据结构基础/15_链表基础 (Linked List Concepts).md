## 16-链表基础 (Linked List Concepts)

链表是一种线性数据结构，元素通过指针链接而非连续内存存储。每个节点包含数据和指向下一个节点的指针。

## 链表 vs 数组

```javascript

// 链表 vs 数组 时间复杂度对比：
//
//           数组    链表
// 访问     O(1)    O(n)
// 查找     O(n)    O(n)
// 头部插入  O(n)    O(1)
// 头部删除  O(n)    O(1)
// 尾部插入  O(1)    O(n)*
// 尾部删除  O(1)    O(n)
// 中间插入  O(n)    O(n)**
//
// * 有尾指针时 O(1)
// ** 找到位置后插入 O(1)

// 链表节点定义
class ListNode {
  constructor(val, next = null) {
    this.val = val;
    this.next = next;
  }
}

// 链表类 (单链表)
class LinkedList {
  constructor() {
    this.head = null;
    this.size = 0;
  }

  // 获取长度
  getSize() { return this.size; }

  // 判断是否为空
  isEmpty() { return this.size === 0; }

  // 头部插入
  addFirst(val) {
    this.head = new ListNode(val, this.head);
    this.size++;
  }

  // 尾部插入
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

  // 打印链表
  print() {
    let curr = this.head;
    let result = [];
    while (curr) {
      result.push(curr.val);
      curr = curr.next;
    }
    result.push('null');
    return result.join(' -> ');
  }
}
```

## 交互演示
