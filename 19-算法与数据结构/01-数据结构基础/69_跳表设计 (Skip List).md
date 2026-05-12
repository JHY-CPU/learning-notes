# 70-跳表设计 (Skip List)

跳表在有序链表上增加多层索引，实现 O(log n) 的平均查找、插入、删除复杂度。Redis 的有序集合（zset）使用跳表实现。

## 原理

```
Level 3:  1 ---------------------------------> 9
Level 2:  1 --------> 5 ---------------------> 9
Level 1:  1 ---> 3 --> 5 --> 7 --------------> 9
Level 0:  1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

从最高层开始查找，如果下一个节点比目标小就右移，否则下降一层
```

## JavaScript 实现

```javascript
class SkipNode {
  constructor(val, level) {
    this.val = val;
    this.next = new Array(level + 1).fill(null);
  }
}

class SkipList {
  constructor(maxLevel = 16, p = 0.5) {
    this.maxLevel = maxLevel;
    this.p = p;
    this.head = new SkipNode(-Infinity, maxLevel);
    this.level = 0;
  }

  _randomLevel() {
    let lvl = 0;
    while (Math.random() < this.p && lvl < this.maxLevel) lvl++;
    return lvl;
  }

  search(target) {
    let curr = this.head;
    for (let i = this.level; i >= 0; i--) {
      while (curr.next[i] && curr.next[i].val < target) {
        curr = curr.next[i];
      }
    }
    curr = curr.next[0];
    return curr !== null && curr.val === target;
  }

  insert(val) {
    const update = new Array(this.maxLevel + 1).fill(null);
    let curr = this.head;

    // 找到每层的前驱节点
    for (let i = this.level; i >= 0; i--) {
      while (curr.next[i] && curr.next[i].val < val) {
        curr = curr.next[i];
      }
      update[i] = curr;
    }

    // 随机决定新节点的层数
    const lvl = this._randomLevel();
    if (lvl > this.level) {
      for (let i = this.level + 1; i <= lvl; i++) {
        update[i] = this.head;
      }
      this.level = lvl;
    }

    // 插入新节点
    const newNode = new SkipNode(val, lvl);
    for (let i = 0; i <= lvl; i++) {
      newNode.next[i] = update[i].next[i];
      update[i].next[i] = newNode;
    }
  }

  remove(val) {
    const update = new Array(this.maxLevel + 1).fill(null);
    let curr = this.head;

    for (let i = this.level; i >= 0; i--) {
      while (curr.next[i] && curr.next[i].val < val) {
        curr = curr.next[i];
      }
      update[i] = curr;
    }

    curr = curr.next[0];
    if (!curr || curr.val !== val) return false;

    // 删除节点
    for (let i = 0; i <= this.level; i++) {
      if (update[i].next[i] !== curr) break;
      update[i].next[i] = curr.next[i];
    }

    // 更新层数
    while (this.level > 0 && this.head.next[this.level] === null) {
      this.level--;
    }
    return true;
  }

  // 范围查询
  range(min, max) {
    let curr = this.head;
    for (let i = this.level; i >= 0; i--) {
      while (curr.next[i] && curr.next[i].val < min) {
        curr = curr.next[i];
      }
    }
    curr = curr.next[0];
    const result = [];
    while (curr && curr.val <= max) {
      result.push(curr.val);
      curr = curr.next[0];
    }
    return result;
  }
}
```

## C++ 实现

```cpp
#include <cstdlib>
#include <vector>
using namespace std;

struct SkipNode {
    int val;
    vector<SkipNode*> next;
    SkipNode(int v, int level) : val(v), next(level + 1, nullptr) {}
};

class SkipList {
    static const int MAX_LEVEL = 16;
    SkipNode* head;
    int level;

public:
    SkipList() : level(0) {
        head = new SkipNode(INT_MIN, MAX_LEVEL);
    }

    bool search(int target) {
        SkipNode* curr = head;
        for (int i = level; i >= 0; i--) {
            while (curr->next[i] && curr->next[i]->val < target)
                curr = curr->next[i];
        }
        curr = curr->next[0];
        return curr && curr->val == target;
    }

    void insert(int val) {
        vector<SkipNode*> update(MAX_LEVEL + 1);
        SkipNode* curr = head;
        for (int i = level; i >= 0; i--) {
            while (curr->next[i] && curr->next[i]->val < val)
                curr = curr->next[i];
            update[i] = curr;
        }
        int lvl = 0;
        while (rand() % 2 == 0 && lvl < MAX_LEVEL) lvl++;
        if (lvl > level) {
            for (int i = level + 1; i <= lvl; i++) update[i] = head;
            level = lvl;
        }
        SkipNode* node = new SkipNode(val, lvl);
        for (int i = 0; i <= lvl; i++) {
            node->next[i] = update[i]->next[i];
            update[i]->next[i] = node;
        }
    }
};
```

## 复杂度分析

| 操作 | 平均 | 最坏 | 空间 |
|------|------|------|------|
| 查找 | O(log n) | O(n) | O(n) |
| 插入 | O(log n) | O(n) | O(n) |
| 删除 | O(log n) | O(n) | O(n) |

## 跳表 vs 平衡树

| 特性 | 跳表 | 红黑树 | B+ 树 |
|------|------|--------|-------|
| 实现复杂度 | 简单 | 复杂 | 复杂 |
| 并发友好 | 是（CAS） | 否 | 部分 |
| 范围查询 | 高效 | 一般 | 高效 |
| 空间开销 | 较多指针 | 较少 | 较少 |
| 使用者 | Redis zset | Java TreeMap | 数据库索引 |

## 应用场景

- **Redis 有序集合**：zset 底层实现
- **LevelDB / RocksDB**：MemTable 实现
- **ConcurrentSkipListMap**：Java 并发有序映射
- **内存数据库**：需要有序且并发的场景

## 常见陷阱

1. **随机层数**：p = 0.5 是常用值，maxLevel 通常 16-32
2. **负无穷头节点**：头节点值为负无穷，方便比较
3. **层间一致性**：insert/remove 时要正确维护每层的前驱
4. **层数更新**：remove 后要清理空的高层
