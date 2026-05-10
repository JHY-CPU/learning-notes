# 链表专题 (Linked List Problems)

## 一、概念定义与原理

### 1.1 链表基础

链表是由节点组成的线性数据结构，每个节点包含数据和指向下一个节点的指针。

**分类：**
- **单链表：** 只有后继指针
- **双链表：** 有前驱和后继指针
- **循环链表：** 尾节点指向头节点

### 1.2 常用技巧

- **快慢指针：** 判环、找中点
- **虚拟头节点：** 简化边界处理
- **递归：** 链表天然适合递归处理

---

## 二、核心算法

### 2.1 快慢指针

- **判环：** fast 每次走两步，slow 每次走一步，若 fast 追上 slow 则有环
- **找中点：** fast 到终点时 slow 到中点
- **找环入口：** 判环后，一个指针从 head 出发，一个从相遇点出发，速度相同，相遇点即为入口

### 2.2 链表翻转

三指针法：prev, curr, next，逐个翻转指针方向。

### 2.3 合并有序链表

类似归并排序的合并过程。

---

## 三、代码实现

### 3.1 链表定义与基础操作 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x): val(x), next(nullptr) {}
};

// 翻转链表
ListNode* reverse(ListNode* head) {
    ListNode *prev = nullptr, *curr = head;
    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// 快慢指针找中点
ListNode* find_middle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
```

### 3.2 判环与找入口 - C++

```cpp
// 判断链表是否有环
bool has_cycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}

// 找环的入口节点
ListNode* detect_cycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            ListNode* p = head;
            while (p != slow) { p = p->next; slow = slow->next; }
            return p;
        }
    }
    return nullptr;
}
```

### 3.3 合并有序链表 - C++

```cpp
ListNode* merge_two_lists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* curr = &dummy;
    while (l1 && l2) {
        if (l1->val <= l2->val) { curr->next = l1; l1 = l1->next; }
        else { curr->next = l2; l2 = l2->next; }
        curr = curr->next;
    }
    curr->next = l1 ? l1 : l2;
    return dummy.next;
}

// 链表排序（归并）
ListNode* sort_list(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* mid = find_middle(head);
    ListNode* right = sort_list(mid->next);
    mid->next = nullptr;
    ListNode* left = sort_list(head);
    return merge_two_lists(left, right);
}
```

### 3.4 Python 实现

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val; self.next = next

def reverse(head):
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False

def merge(l1, l2):
    dummy = ListNode(0); curr = dummy
    while l1 and l2:
        if l1.val <= l2.val: curr.next = l1; l1 = l1.next
        else: curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### 3.5 删除倒数第N个节点

```cpp
ListNode* remove_nth_from_end(ListNode* head, int n) {
    ListNode dummy(0); dummy.next = head;
    ListNode *fast = &dummy, *slow = &dummy;
    for (int i = 0; i <= n; i++) fast = fast->next;
    while (fast) { fast = fast->next; slow = slow->next; }
    slow->next = slow->next->next;
    return dummy.next;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 翻转链表 | $O(n)$ | $O(1)$ |
| 找中点 | $O(n)$ | $O(1)$ |
| 判环 | $O(n)$ | $O(1)$ |
| 合并两链表 | $O(n+m)$ | $O(1)$ |
| 链表排序 | $O(n \log n)$ | $O(\log n)$ 递归栈 |

---

## 五、竞赛与面试应用场景

1. **LeetCode 206：** 翻转链表
2. **LeetCode 141：** 环形链表
3. **LeetCode 21：** 合并两个有序链表
4. **LeetCode 148：** 链表排序
5. **LeetCode 19：** 删除倒数第N个节点
