# 46-哈希表实战应用 (Hash Table Applications)

哈希表是工程实践中使用最广泛的数据结构之一，几乎所有需要快速查找的场景都有它的身影。

## 缓存系统

```javascript
// 带 TTL 的缓存
class TTLCache {
  constructor(ttl = 5000) {
    this.cache = new Map();
    this.ttl = ttl;
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expire) {
      this.cache.delete(key);
      return null;
    }
    return entry.value;
  }

  set(key, value) {
    this.cache.set(key, { value, expire: Date.now() + this.ttl });
  }

  delete(key) { this.cache.delete(key); }
  size() { return this.cache.size; }
}
```

## 频率统计

```javascript
// 词频统计
function wordFrequency(text) {
  const freq = new Map();
  for (const word of text.toLowerCase().split(/\s+/)) {
    freq.set(word, (freq.get(word) || 0) + 1);
  }
  return freq;
}

// 前 K 高频元素
function topKFrequent(nums, k) {
  const freq = new Map();
  for (const n of nums) freq.set(n, (freq.get(n) || 0) + 1);
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(e => e[0]);
}
```

## C++ 实现

```cpp
#include <unordered_map>
#include <string>
#include <vector>
using namespace std;

// 符号表
class SymbolTable {
    unordered_map<string, int> table;
public:
    void define(const string& name, int val) { table[name] = val; }
    int lookup(const string& name) { return table.count(name) ? table[name] : -1; }
    bool exists(const string& name) { return table.count(name) > 0; }
    void remove(const string& name) { table.erase(name); }
};

// 双射（双向映射）
class BiMap {
    unordered_map<int, string> forward;
    unordered_map<string, int> backward;
public:
    void insert(int key, const string& val) {
        forward[key] = val;
        backward[val] = key;
    }
    string getByKey(int key) { return forward.count(key) ? forward[key] : ""; }
    int getByVal(const string& val) { return backward.count(val) ? backward[val] : -1; }
};
```

## 实际应用场景

```javascript
// 1. 路由表
class Router {
  constructor() {
    this.routes = new Map();
  }
  register(path, handler) { this.routes.set(path, handler); }
  dispatch(path) {
    const handler = this.routes.get(path);
    if (!handler) return '404 Not Found';
    return handler();
  }
}

// 2. 依赖注入容器
class DIContainer {
  constructor() {
    this.services = new Map();
    this.instances = new Map();
  }
  register(name, factory) { this.services.set(name, factory); }
  resolve(name) {
    if (!this.instances.has(name)) {
      const factory = this.services.get(name);
      if (!factory) throw new Error(`Service ${name} not found`);
      this.instances.set(name, factory(this));
    }
    return this.instances.get(name);
  }
}

// 3. 事件发射器
class EventEmitter {
  constructor() {
    this.listeners = new Map();
  }
  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
  }
  emit(event, ...args) {
    const fns = this.listeners.get(event) || [];
    fns.forEach(fn => fn(...args));
  }
  off(event, fn) {
    const fns = this.listeners.get(event);
    if (fns) this.listeners.set(event, fns.filter(f => f !== fn));
  }
}
```

## 哈希表适用场景

| 场景 | 哈希表方案 | 替代方案 |
|------|-----------|---------|
| 精确查找 | Map O(1) | 数组 O(n)，BST O(log n) |
| 缓存 | Map + TTL | Redis, Memcached |
| 去重 | Set O(1) | 排序 O(n log n) |
| 计数 | Map | 数组（范围已知） |
| 前缀匹配 | - | Trie 树 |
| 范围查询 | - | B+ 树，跳表 |

## 常见陷阱

1. **缓存穿透**：查询不存在的键频繁穿透到后端
2. **缓存雪崩**：大量键同时过期导致后端压力激增
3. **内存泄漏**：只增不删的哈希表会持续增长
4. **并发安全**：多线程环境下需要加锁或使用并发哈希表
