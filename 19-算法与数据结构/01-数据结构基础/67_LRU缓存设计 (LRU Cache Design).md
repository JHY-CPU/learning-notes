# 68-LRU缓存设计 (LRU Cache Design)

LRU（Least Recently Used）缓存淘汰最近最少使用的数据，核心是哈希表+双向链表。

## 实现原理

- **哈希表**：O(1) 查找节点
- **双向链表**：O(1) 移动节点到头部 / 删除尾部
- 访问（get/put）时将节点移到链表头部
- 容量满时删除链表尾部（最久未使用）

## JavaScript 实现

```javascript
class LRUNode {
  constructor(key, val) {
    this.key = key;
    this.val = val;
    this.prev = null;
    this.next = null;
  }
}

class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.map = new Map(); // key -> node
    // 虚拟头尾节点简化边界处理
    this.head = new LRUNode(0, 0);
    this.tail = new LRUNode(0, 0);
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  get(key) {
    if (!this.map.has(key)) return -1;
    const node = this.map.get(key);
    this._remove(node);
    this._addToHead(node);
    return node.val;
  }

  put(key, value) {
    if (this.map.has(key)) {
      const node = this.map.get(key);
      node.val = value;
      this._remove(node);
      this._addToHead(node);
    } else {
      const node = new LRUNode(key, value);
      this.map.set(key, node);
      this._addToHead(node);
      if (this.map.size > this.capacity) {
        const lru = this.tail.prev;
        this._remove(lru);
        this.map.delete(lru.key);
      }
    }
  }

  _remove(node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
  }

  _addToHead(node) {
    node.next = this.head.next;
    node.prev = this.head;
    this.head.next.prev = node;
    this.head.next = node;
  }
}

// 使用
const cache = new LRUCache(2);
cache.put(1, 1);
cache.put(2, 2);
console.log(cache.get(1));    // 1
cache.put(3, 3);              // 淘汰 key=2
console.log(cache.get(2));    // -1
console.log(cache.get(3));    // 3
```

## C++ 实现

```cpp
#include <unordered_map>
#include <list>
using namespace std;

class LRUCache {
    int cap;
    list<pair<int, int>> cache; // {key, value}
    unordered_map<int, list<pair<int,int>>::iterator> map;

public:
    LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        if (!map.count(key)) return -1;
        // 移到头部
        auto it = map[key];
        cache.splice(cache.begin(), cache, it);
        return it->second;
    }

    void put(int key, int value) {
        if (map.count(key)) {
            auto it = map[key];
            it->second = value;
            cache.splice(cache.begin(), cache, it);
        } else {
            cache.emplace_front(key, value);
            map[key] = cache.begin();
            if ((int)cache.size() > cap) {
                auto last = cache.back();
                map.erase(last.first);
                cache.pop_back();
            }
        }
    }
};
```

## 简化实现（利用 JS Map 特性）

```javascript
// JavaScript Map 保持插入顺序，可简化实现
class SimpleLRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key) {
    if (!this.cache.has(key)) return -1;
    const val = this.cache.get(key);
    // 移到最后（最近使用）
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }

  put(key, value) {
    if (this.cache.has(key)) this.cache.delete(key);
    else if (this.cache.size >= this.capacity) {
      // 删除最早的（Map 的第一个键）
      this.cache.delete(this.cache.keys().next().value);
    }
    this.cache.set(key, value);
  }
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| get | O(1) | - |
| put | O(1) | - |
| 空间 | O(capacity) | - |

## LRU vs 其他缓存策略

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| LRU | 淘汰最久未使用 | 通用缓存 |
| LFU | 淘汰使用频率最低 | 频率敏感场景 |
| FIFO | 淘汰最早进入 | 简单队列 |
| Random | 随机淘汰 | 简单场景 |

## 应用场景

- **操作系统页面置换**：虚拟内存管理
- **数据库缓存**：Buffer Pool 页面淘汰
- **CDN 缓存**：热点内容缓存
- **浏览器缓存**：HTTP 缓存淘汰
- **CPU 缓存**：Cache 替换策略

## 常见陷阱

1. **get 也要更新**：get 操作也要将节点移到最新
2. **虚拟头尾**：使用虚拟头尾节点简化边界判断
3. **线程安全**：多线程环境需要加锁
4. **容量为 0**：特殊处理容量为 0 的情况
