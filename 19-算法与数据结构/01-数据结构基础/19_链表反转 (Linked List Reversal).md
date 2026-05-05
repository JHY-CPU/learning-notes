## 20-链表反转 (Linked List Reversal)

链表反转是一类核心操作，包括全反转、区间反转、K个一组反转等变体。

## 反转整个链表

```javascript

// 迭代法（三指针）
function reverseList(head) {
  let prev = null;
  let curr = head;

  while (curr) {
    let next = curr.next; // 保存下一个节点
    curr.next = prev;     // 指向前一个节点
    prev = curr;          // 前移 prev
    curr = next;          // 前移 curr
  }

  return prev; // 新头节点
}

// 递归法
function reverseListRecursive(head) {
  if (!head || !head.next) return head;

  let newHead = reverseListRecursive(head.next);
  head.next.next = head;
  head.next = null;

  return newHead;
}
```

## 区间反转

```javascript

// 反转链表中从 left 到 right 的部分（1-based）
function reverseBetween(head, left, right) {
  let dummy = new ListNode(null);
  dummy.next = head;
  let prev = dummy;

  // prev 移动到 left 前一个节点
  for (let i = 1; i < left; i++) {
    prev = prev.next;
  }

  // 开始反转：头插法
  let curr = prev.next;
  for (let i = 0; i < right - left; i++) {
    let next = curr.next;
    curr.next = next.next;
    next.next = prev.next;
    prev.next = next;
  }

  return dummy.next;
}

// K个一组反转
function reverseKGroup(head, k) {
  let dummy = new ListNode(null);
  dummy.next = head;
  let prev = dummy;

  while (true) {
    // 检查剩余节点是否够 k 个
    let end = prev;
    for (let i = 0; i < k && end; i++) end = end.next;
    if (!end) break;

    // 头插法反转 k 个节点
    let curr = prev.next;
    for (let i = 0; i < k - 1; i++) {
      let next = curr.next;
      curr.next = next.next;
      next.next = prev.next;
      prev.next = next;
    }

    prev = curr; // 移动 prev 到下一组的前一个节点
  }

  return dummy.next;
}
```

## 交互演示
