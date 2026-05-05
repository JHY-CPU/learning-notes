## 18-双向链表 (Doubly Linked List)

双向链表的每个节点包含 prev 和 next 两个指针，可以从两个方向遍历。插入和删除操作比单链表更灵活。

## 双向链表节点与类

```javascript

// 双向链表节点
class DoublyListNode {
  constructor(val) {
    this.val = val;
    this.prev = null;
    this.next = null;
  }
}

// 双向链表类（带虚拟头尾节点）
class DoublyLinkedList {
  constructor() {
    this.dummyHead = new DoublyListNode(null);
    this.dummyTail = new DoublyListNode(null);
    // 连接虚拟头尾
    this.dummyHead.next = this.dummyTail;
    this.dummyTail.prev = this.dummyHead;
    this.size = 0;
  }

  // 在指定节点的后面插入
  addAfter(node, val) {
    let newNode = new DoublyListNode(val);
    let nextNode = node.next;

    // 建立新节点的连接
    newNode.prev = node;
    newNode.next = nextNode;

    // 更新前后节点的连接
    node.next = newNode;
    nextNode.prev = newNode;

    this.size++;
    return newNode;
  }

  // 删除指定节点
  removeNode(node) {
    if (node === this.dummyHead || node === this.dummyTail) return;

    let prev = node.prev;
    let next = node.next;
    prev.next = next;
    next.prev = prev;

    node.prev = null;
    node.next = null;
    this.size--;
  }

  // 头部插入
  addFirst(val) {
    return this.addAfter(this.dummyHead, val);
  }

  // 尾部插入
  addLast(val) {
    return this.addAfter(this.dummyTail.prev, val);
  }

  // 正向遍历
  forward() {
    let result = [];
    let curr = this.dummyHead.next;
    while (curr !== this.dummyTail) {
      result.push(curr.val);
      curr = curr.next;
    }
    return result;
  }

  // 反向遍历
  backward() {
    let result = [];
    let curr = this.dummyTail.prev;
    while (curr !== this.dummyHead) {
      result.push(curr.val);
      curr = curr.prev;
    }
    return result;
  }
}
```

## 交互演示
