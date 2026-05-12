# 17-单链表实现 (Singly Linked List Implementation)

单链表的完整实现，包括增删改查、按索引操作等核心方法。

## 完整单链表实现（JavaScript）

```javascript
class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

class SinglyLinkedList {
  constructor() {
    this.dummyHead = new ListNode(null); // 虚拟头节点简化边界
    this.size = 0;
  }

  getNode(index) {
    if (index < 0 || index >= this.size) throw new Error('越界');
    let curr = this.dummyHead.next;
    for (let i = 0; i < index; i++) curr = curr.next;
    return curr;
  }

  get(index) { return this.getNode(index).val; }

  addAtIndex(index, val) {
    if (index < 0 || index > this.size) return;
    let prev = this.dummyHead;
    for (let i = 0; i < index; i++) prev = prev.next;
    let newNode = new ListNode(val);
    newNode.next = prev.next;
    prev.next = newNode;
    this.size++;
  }

  removeAtIndex(index) {
    if (index < 0 || index >= this.size) throw new Error('越界');
    let prev = this.dummyHead;
    for (let i = 0; i < index; i++) prev = prev.next;
    let removed = prev.next;
    prev.next = removed.next;
    removed.next = null;
    this.size--;
    return removed.val;
  }

  addFirst(val) { this.addAtIndex(0, val); }
  addLast(val) { this.addAtIndex(this.size, val); }

  toArray() {
    let result = [];
    let curr = this.dummyHead.next;
    while (curr) { result.push(curr.val); curr = curr.next; }
    return result;
  }
}
```

## C++ 完整实现

```cpp
#include <iostream>
#include <stdexcept>
using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class SinglyLinkedList {
    ListNode* dummyHead;
    int sz;

public:
    SinglyLinkedList() : sz(0) {
        dummyHead = new ListNode(0);
    }

    ~SinglyLinkedList() {
        ListNode* curr = dummyHead;
        while (curr) {
            ListNode* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    int get(int index) {
        if (index < 0 || index >= sz) throw out_of_range("index");
        ListNode* curr = dummyHead->next;
        for (int i = 0; i < index; i++) curr = curr->next;
        return curr->val;
    }

    void addAtHead(int val) {
        ListNode* node = new ListNode(val);
        node->next = dummyHead->next;
        dummyHead->next = node;
        sz++;
    }

    void addAtTail(int val) {
        ListNode* curr = dummyHead;
        while (curr->next) curr = curr->next;
        curr->next = new ListNode(val);
        sz++;
    }

    void addAtIndex(int index, int val) {
        if (index < 0 || index > sz) return;
        ListNode* prev = dummyHead;
        for (int i = 0; i < index; i++) prev = prev->next;
        ListNode* node = new ListNode(val);
        node->next = prev->next;
        prev->next = node;
        sz++;
    }

    void deleteAtIndex(int index) {
        if (index < 0 || index >= sz) return;
        ListNode* prev = dummyHead;
        for (int i = 0; i < index; i++) prev = prev->next;
        ListNode* toDelete = prev->next;
        prev->next = toDelete->next;
        delete toDelete;
        sz--;
    }

    int size() const { return sz; }

    void print() {
        ListNode* curr = dummyHead->next;
        while (curr) {
            cout << curr->val;
            if (curr->next) cout << " -> ";
            curr = curr->next;
        }
        cout << " -> null" << endl;
    }
};
```

## 虚拟头节点技巧

使用虚拟头节点（dummy node）可以统一处理头部插入和删除的边界情况：

```javascript
// 不用虚拟头节点：需要特殊处理 head
function deleteNode(head, val) {
  if (head.val === val) return head.next; // 特殊处理
  let curr = head;
  while (curr.next) {
    if (curr.next.val === val) {
      curr.next = curr.next.next;
      return head;
    }
    curr = curr.next;
  }
  return head;
}

// 用虚拟头节点：统一逻辑
function deleteNode(dummyHead, val) {
  let prev = dummyHead;
  while (prev.next) {
    if (prev.next.val === val) {
      prev.next = prev.next.next;
      return;
    }
    prev = prev.next;
  }
}
```

## 常用操作速查

| 方法 | 时间复杂度 |
|------|-----------|
| get(index) | O(n) |
| addAtHead | O(1) |
| addAtTail | O(n) |
| addAtIndex | O(n) |
| deleteAtIndex | O(n) |

## 常见陷阱

1. **边界检查**：index 的有效范围是 [0, size)
2. **虚拟头节点**：统一头尾操作，减少边界判断
3. **遍历条件**：`curr` vs `curr.next` 的使用场景
4. **内存管理**：C++ 中删除节点后需要 delete
