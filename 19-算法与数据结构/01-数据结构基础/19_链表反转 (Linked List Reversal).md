# 20-链表反转 (Linked List Reversal)

链表反转是一类核心操作，包括全反转、区间反转、K个一组反转等变体。

## 反转整个链表

```javascript
// 迭代法（三指针）
function reverseList(head) {
  let prev = null, curr = head;
  while (curr) {
    let next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }
  return prev;
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

## C++ 实现

```cpp
// 迭代
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// 递归
ListNode* reverseRecursive(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverseRecursive(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

## 区间反转（头插法）

```javascript
// 反转链表中从 left 到 right 的部分（1-based）
function reverseBetween(head, left, right) {
  let dummy = new ListNode(null);
  dummy.next = head;
  let prev = dummy;

  for (let i = 1; i < left; i++) prev = prev.next;

  let curr = prev.next;
  for (let i = 0; i < right - left; i++) {
    let next = curr.next;
    curr.next = next.next;
    next.next = prev.next;
    prev.next = next;
  }
  return dummy.next;
}
```

## K 个一组反转

```javascript
function reverseKGroup(head, k) {
  // 检查是否有 k 个节点
  let end = head;
  for (let i = 0; i < k; i++) {
    if (!end) return head;
    end = end.next;
  }

  // 反转前 k 个
  let prev = null, curr = head;
  for (let i = 0; i < k; i++) {
    let next = curr.next;
    curr.next = prev;
    prev = curr;
    curr = next;
  }

  // head 现在是反转后的尾部，递归处理剩余部分
  head.next = reverseKGroup(curr, k);
  return prev; // prev 是反转后的头部
}
```

## 两两交换节点

```javascript
function swapPairs(head) {
  let dummy = new ListNode(0, head);
  let prev = dummy;
  while (prev.next && prev.next.next) {
    let a = prev.next;
    let b = prev.next.next;
    a.next = b.next;
    b.next = a;
    prev.next = b;
    prev = a;
  }
  return dummy.next;
}
```

## 反转链表的常见变体

```javascript
// 1. 反转前 N 个节点
function reverseN(head, n) {
  if (n === 1) return head;
  let successor = null;

  function reverse(head, n) {
    if (n === 1) {
      successor = head.next;
      return head;
    }
    let last = reverse(head.next, n - 1);
    head.next.next = head;
    head.next = successor;
    return last;
  }
  return reverse(head, n);
}

// 2. 回文链表判断（反转后半部分比较）
function isPalindrome(head) {
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  let prev = null;
  while (slow) {
    let next = slow.next;
    slow.next = prev;
    prev = slow;
    slow = next;
  }
  let p1 = head, p2 = prev;
  while (p2) {
    if (p1.val !== p2.val) return false;
    p1 = p1.next;
    p2 = p2.next;
  }
  return true;
}
```

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 迭代反转 | O(n) | O(1) |
| 递归反转 | O(n) | O(n) 栈 |
| 区间反转 | O(n) | O(1) |
| K组反转 | O(n) | O(n/k) 栈 |

## 关键技巧

1. **虚拟头节点**：处理 left = 1 的边界
2. **头插法**：区间反转的核心，每次将当前节点插到区间头部
3. **递归理解**：假设后续已反转，只需处理当前节点
4. **保存引用**：修改指针前一定要保存后续节点

## 常见陷阱

1. 丢失 next 引用导致链表断裂
2. 递归法中 head.next = null 容易忘记
3. 区间反转中循环次数为 right - left 次
4. K 组反转中不足 k 个的剩余节点不反转
