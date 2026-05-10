# 链表专题 (Linked List Problems)

## 一、概念定义与原理

### 1.1 链表基础

链表是由节点组成的线性数据结构，每个节点包含数据域和指针域。

**链表分类：**
- **单链表：** 只有后继指针 `next`
- **双向链表：** 有前驱 `prev` 和后继 `next` 指针
- **循环链表：** 尾节点的 `next` 指向头节点

**与数组对比：**

| 操作 | 数组 | 链表 |
|------|------|------|
| 随机访问 | $O(1)$ | $O(n)$ |
| 头部插入 | $O(n)$ | $O(1)$ |
| 中间插入 | $O(n)$ | $O(1)$（已找到位置） |
| 内存 | 连续 | 分散 |

### 1.2 链表节点定义

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
```

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

---

## 二、核心技巧

### 2.1 虚拟头节点 (Dummy Node)

解决头节点可能被删除或修改的问题，简化边界处理。

```python
def remove_elements(head, val):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    while prev.next:
        if prev.next.val == val:
            prev.next = prev.next.next
        else:
            prev = prev.next
    return dummy.next
```

### 2.2 快慢指针

**找中点：**
```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

**判环：**
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

**找环入口：** 判环后，一个指针从 head 出发，一个从相遇点出发，速度相同，相遇点即为入口。

### 2.3 链表翻转

**迭代法：**
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

**递归法：**
```python
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

---

## 三、经典题目详解

### 3.1 合并K个有序链表 (LeetCode 23)

```python
import heapq

def mergeKLists(lists):
    dummy = ListNode(0)
    curr = dummy
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

### 3.2 两数相加 (LeetCode 2)

```python
def add_two_numbers(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    while l1 or l2 or carry:
        val = carry
        if l1: val += l1.val; l1 = l1.next
        if l2: val += l2.val; l2 = l2.next
        carry, val = divmod(val, 10)
        curr.next = ListNode(val)
        curr = curr.next
    return dummy.next
```

### 3.3 排序链表 (LeetCode 148)

**归并排序：** 找中点、递归分割、合并。

```python
def sort_list(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    left = sort_list(head)
    right = sort_list(mid)
    return merge(left, right)

def merge(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1; l1 = l1.next
        else:
            curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### 3.4 反转链表II (LeetCode 92)

```cpp
ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode dummy(0);
    dummy.next = head;
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

### 3.5 回文链表 (LeetCode 234)

```python
def is_palindrome(head):
    # 找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # 翻转后半部分
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
    # 比较
    p1, p2 = head, prev
    while p2:
        if p1.val != p2.val: return False
        p1 = p1.next
        p2 = p2.next
    return True
```

---

## 四、C++ 完整实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(nullptr) {}
};

// 删除倒数第N个节点
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0); dummy.next = head;
    ListNode *fast = &dummy, *slow = &dummy;
    for (int i = 0; i <= n; i++) fast = fast->next;
    while (fast) { fast = fast->next; slow = slow->next; }
    slow->next = slow->next->next;
    return dummy.next;
}

// 交叉链表找交点
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *a = headA, *b = headB;
    while (a != b) {
        a = a ? a->next : headB;
        b = b ? b->next : headA;
    }
    return a;
}

// 奇偶链表分组
ListNode* oddEvenList(ListNode* head) {
    if (!head) return nullptr;
    ListNode *odd = head, *even = head->next, *evenHead = even;
    while (even && even->next) {
        odd->next = even->next;
        odd = odd->next;
        even->next = odd->next;
        even = even->next;
    }
    odd->next = evenHead;
    return head;
}
```

---

## 五、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 翻转链表 | $O(n)$ | $O(1)$ |
| 找中点 | $O(n)$ | $O(1)$ |
| 判环 | $O(n)$ | $O(1)$ |
| 合并K链表(堆) | $O(N \log k)$ | $O(k)$ |
| 排序链表 | $O(n \log n)$ | $O(\log n)$ |
| 回文判断 | $O(n)$ | $O(1)$ |

---

## 六、面试高频题

1. **LeetCode 206：** 反转链表
2. **LeetCode 21：** 合并两个有序链表
3. **LeetCode 141/142：** 环形链表 I/II
4. **LeetCode 19：** 删除倒数第N个节点
5. **LeetCode 23：** 合并K个有序链表
6. **LeetCode 25：** K个一组翻转链表
7. **LeetCode 148：** 排序链表
8. **LeetCode 160：** 相交链表
9. **LeetCode 234：** 回文链表
10. **LeetCode 146：** LRU缓存（哈希+双向链表）
