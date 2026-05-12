# 19-循环链表 (Circular Linked List)

循环链表是链表的变体，尾节点的 next 指向头节点，形成环状结构。可以是单向或双向。

## 循环链表特点

1. 尾节点的 next 指向头节点（而非 null）
2. 从任意节点出发都能遍历整个链表
3. 适合需要反复循环遍历的场景
4. 经典应用：约瑟夫问题、轮转调度

## JavaScript 实现

```javascript
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

  // 尾部插入 O(1)
  addLast(val) {
    let newNode = new CircularListNode(val);
    if (!this.head) {
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

  // 删除指定值的节点 O(n)
  remove(val) {
    if (!this.head) return false;
    let curr = this.head, prev = this.tail;
    do {
      if (curr.val === val) {
        if (this.size === 1) {
          this.head = null; this.tail = null;
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

  // 遍历指定步数
  traverse(steps) {
    if (!this.head) return [];
    let result = [], curr = this.head;
    for (let i = 0; i < steps; i++) {
      result.push(curr.val);
      curr = curr.next;
    }
    return result;
  }
}
```

## C++ 实现

```cpp
struct CListNode {
    int val;
    CListNode* next;
    CListNode(int x) : val(x), next(nullptr) {}
};

class CircularList {
    CListNode* head;
    CListNode* tail;
    int sz;
public:
    CircularList() : head(nullptr), tail(nullptr), sz(0) {}

    void addLast(int val) {
        CListNode* node = new CListNode(val);
        if (!head) {
            head = tail = node;
            node->next = node;
        } else {
            node->next = head;
            tail->next = node;
            tail = node;
        }
        sz++;
    }

    int removeAt(int index) {
        if (sz == 0) throw runtime_error("empty");
        index %= sz;
        CListNode* prev = tail;
        CListNode* curr = head;
        for (int i = 0; i < index; i++) {
            prev = curr;
            curr = curr->next;
        }
        int val = curr->val;
        if (curr == head) head = head->next;
        if (curr == tail) tail = prev;
        prev->next = curr->next;
        delete curr;
        sz--;
        return val;
    }
};
```

## 约瑟夫问题

```javascript
// n 个人围成圈，从第 k 个人开始报数，每数到 m 的人出列
// 求最后剩下的人的编号
function josephus(n, m) {
  let head = { val: 0, next: null };
  let curr = head;
  for (let i = 1; i < n; i++) {
    curr.next = { val: i, next: null };
    curr = curr.next;
  }
  curr.next = head; // 形成环

  let prev = curr;
  curr = head;
  while (curr.next !== curr) {
    for (let i = 0; i < m - 1; i++) {
      prev = curr;
      curr = curr.next;
    }
    prev.next = curr.next;
    curr = prev.next;
  }
  return curr.val;
}

// 数学解法 O(n)
function josephusMath(n, m) {
  let pos = 0;
  for (let i = 2; i <= n; i++) {
    pos = (pos + m) % i;
  }
  return pos;
}
```

## 应用场景

- **操作系统轮转调度**：CPU 时间片轮转
- **游戏回合制**：玩家轮流行动
- **环形缓冲区**：数据流处理
- **音乐播放列表循环**：循环播放
- **约瑟夫问题**：经典面试题

## 循环链表 vs 普通链表

| 特性 | 普通链表 | 循环链表 |
|------|---------|---------|
| 尾节点 | next = null | next = head |
| 遍历终止 | curr !== null | curr !== head |
| 从尾到头 | 不可直接访问 | O(1) |
| 典型应用 | 通用 | 约瑟夫、轮转 |

## 常见陷阱

1. **无限循环**：遍历时必须有终止条件
2. **空链表**：处理 size = 0 的特殊情况
3. **单节点删除**：删除后要正确处理自环
4. **边界更新**：删除头/尾节点时要更新 head/tail
