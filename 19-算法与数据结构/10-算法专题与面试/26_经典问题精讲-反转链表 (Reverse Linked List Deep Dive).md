# 经典问题精讲-反转链表 (Reverse Linked List Deep Dive)

## 一、问题系列

反转链表是链表问题的核心，有多个变种：

| 题目 | 难度 | 要点 |
|------|------|------|
| 反转链表 (LC 206) | Easy | 基础反转 |
| 反转链表II (LC 92) | Medium | 区间反转 |
| K个一组翻转 (LC 25) | Hard | 分组反转 |
| 两两交换节点 (LC 24) | Medium | K=2的特例 |
| 反转链表前N个节点 | — | 递归变种 |

---

## 二、基础：反转整个链表 (LeetCode 206)

### 2.1 迭代法 — 三指针

```python
def reverse_list(head):
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

**图解过程：**
```
初始: 1 -> 2 -> 3 -> 4 -> None

第1步: None <- 1    2 -> 3 -> 4 -> None
第2步: None <- 1 <- 2    3 -> 4 -> None
第3步: None <- 1 <- 2 <- 3    4 -> None
第4步: None <- 1 <- 2 <- 3 <- 4

结果: 4 -> 3 -> 2 -> 1 -> None
```

### 2.2 递归法

```python
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head  # 反转指向
    head.next = None       # 断开原指向
    return new_head
```

**递归理解：**
```
reverse(1->2->3->4):
  new_head = reverse(2->3->4) -> 返回 4->3->2
  然后: 2.next = 1, 所以 4->3->2->1
  1.next = None
```

---

## 三、进阶：反转链表II (LeetCode 92)

**反转从位置 left 到 right 的部分。**

```python
def reverse_between(head, left, right):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    # 移动到 left 前一个位置
    for _ in range(left - 1):
        prev = prev.next

    curr = prev.next
    # 头插法：将 curr 后面的节点依次插到 prev 后面
    for _ in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

**头插法图解：** 反转 [2,3,4] 区间
```
初始: 1 -> 2 -> 3 -> 4 -> 5
      prev curr

第1次: 1 -> 3 -> 2 -> 4 -> 5  (将3插到prev后)
第2次: 1 -> 4 -> 3 -> 2 -> 5  (将4插到prev后)

完成！
```

---

## 四、高级：K个一组翻转链表 (LeetCode 25)

```python
def reverse_k_group(head, k):
    # 检查是否有k个节点
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1
    if count < k:
        return head

    # 反转前k个
    prev, curr = None, head
    for _ in range(k):
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    # head 现在是反转后的尾部，连接下一段
    head.next = reverse_k_group(curr, k)

    return prev  # prev 是反转后的头部
```

---

## 五、两两交换节点 (LeetCode 24)

```python
def swap_pairs(head):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while prev.next and prev.next.next:
        a = prev.next
        b = prev.next.next

        # 交换
        a.next = b.next
        b.next = a
        prev.next = b

        prev = a

    return dummy.next
```

---

## 六、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(nullptr) {}
};

// 反转整个链表
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* nxt = curr->next;
        curr->next = prev;
        prev = curr;
        curr = nxt;
    }
    return prev;
}

// 反转区间
ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode dummy(0); dummy.next = head;
    ListNode* prev = &dummy;
    for (int i = 1; i < left; i++) prev = prev->next;
    ListNode* curr = prev->next;
    for (int i = 0; i < right - left; i++) {
        ListNode* next = curr->next;
        curr->next = next->next;
        next->next = prev->next;
        prev->next = next;
    }
    return dummy.next;
}
```

---

## 七、复杂度分析

| 算法 | 时间 | 空间 |
|------|------|------|
| 反转链表(迭代) | $O(n)$ | $O(1)$ |
| 反转链表(递归) | $O(n)$ | $O(n)$ 栈 |
| 反转区间 | $O(n)$ | $O(1)$ |
| K组翻转 | $O(n)$ | $O(n/k)$ 栈 |
