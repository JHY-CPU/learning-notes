# 44-哈希表与设计 (Hash Table Design)

设计哈希表需要考虑哈希函数、冲突解决、动态扩容和负载因子四个核心要素。

## 完整哈希表实现

```javascript
class HashTable {
  constructor(capacity = 8, loadFactor = 0.75) {
    this.capacity = capacity;
    this.loadFactor = loadFactor;
    this.size = 0;
    this.buckets = new Array(capacity).fill(null).map(() => []);
  }

  // 哈希函数
  _hash(key) {
    let hash = 0;
    const str = String(key);
    for (let i = 0; i < str.length; i++) {
      hash = (hash * 31 + str.charCodeAt(i)) >>> 0; // 无符号
    }
    return hash % this.capacity;
  }

  // 插入/更新
  set(key, value) {
    const idx = this._hash(key);
    const bucket = this.buckets[idx];
    for (const pair of bucket) {
      if (pair[0] === key) { pair[1] = value; return; }
    }
    bucket.push([key, value]);
    this.size++;
    if (this.size > this.capacity * this.loadFactor) this._resize();
  }

  // 查找
  get(key) {
    const idx = this._hash(key);
    for (const [k, v] of this.buckets[idx]) {
      if (k === key) return v;
    }
    return undefined;
  }

  // 删除
  delete(key) {
    const idx = this._hash(key);
    const bucket = this.buckets[idx];
    for (let i = 0; i < bucket.length; i++) {
      if (bucket[i][0] === key) {
        bucket.splice(i, 1);
        this.size--;
        return true;
      }
    }
    return false;
  }

  has(key) { return this.get(key) !== undefined; }

  // 扩容
  _resize() {
    const old = this.buckets;
    this.capacity *= 2;
    this.buckets = new Array(this.capacity).fill(null).map(() => []);
    this.size = 0;
    for (const bucket of old) {
      for (const [k, v] of bucket) this.set(k, v);
    }
  }

  keys() {
    const result = [];
    for (const bucket of this.buckets) {
      for (const [k] of bucket) result.push(k);
    }
    return result;
  }

  values() {
    const result = [];
    for (const bucket of this.buckets) {
      for (const [, v] of bucket) result.push(v);
    }
    return result;
  }
}
```

## C++ 实现

```cpp
#include <vector>
#include <list>
#include <string>
using namespace std;

class HashTable {
    vector<list<pair<string, int>>> buckets;
    int cap;
    int sz;
    double lf;

    int hash(const string& key) {
        unsigned h = 0;
        for (char c : key) h = h * 31 + c;
        return h % cap;
    }

    void resize() {
        auto old = move(buckets);
        cap *= 2;
        buckets.assign(cap, {});
        sz = 0;
        for (auto& bucket : old) {
            for (auto& [k, v] : bucket) set(k, v);
        }
    }

public:
    HashTable(int c = 8) : cap(c), sz(0), lf(0.75) {
        buckets.resize(cap);
    }

    void set(const string& key, int val) {
        int idx = hash(key);
        for (auto& [k, v] : buckets[idx]) {
            if (k == key) { v = val; return; }
        }
        buckets[idx].emplace_back(key, val);
        if (++sz > cap * lf) resize();
    }

    int get(const string& key, int def = -1) {
        int idx = hash(key);
        for (auto& [k, v] : buckets[idx]) {
            if (k == key) return v;
        }
        return def;
    }

    bool remove(const string& key) {
        int idx = hash(key);
        auto& bucket = buckets[idx];
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->first == key) { bucket.erase(it); sz--; return true; }
        }
        return false;
    }
};
```

## 冲突解决策略

```javascript
// 1. 链地址法（上面实现的就是）
// 每个桶维护一个链表，冲突元素追加到链表

// 2. 开放地址法
class OpenAddressingHashTable {
  constructor(capacity = 16) {
    this.capacity = capacity;
    this.keys = new Array(capacity).fill(null);
    this.values = new Array(capacity).fill(null);
    this.size = 0;
  }

  _hash(key) {
    let h = 0;
    for (const ch of String(key)) h = (h * 31 + ch.charCodeAt(0)) % this.capacity;
    return h;
  }

  set(key, value) {
    let idx = this._hash(key);
    while (this.keys[idx] !== null && this.keys[idx] !== key) {
      idx = (idx + 1) % this.capacity; // 线性探测
    }
    if (this.keys[idx] === null) this.size++;
    this.keys[idx] = key;
    this.values[idx] = value;
  }

  get(key) {
    let idx = this._hash(key);
    while (this.keys[idx] !== null) {
      if (this.keys[idx] === key) return this.values[idx];
      idx = (idx + 1) % this.capacity;
    }
    return undefined;
  }
}
```

## 设计要点

| 要素 | 说明 | 推荐值 |
|------|------|--------|
| 哈希函数 | 均匀分布、计算快 | 乘法哈希、MurmurHash |
| 冲突解决 | 链地址 vs 开放地址 | 链地址更通用 |
| 负载因子 | n/m 的阈值 | 0.75 |
| 扩容策略 | 容量翻倍 + rehash | 2x 扩容 |
| 初始容量 | 避免频繁扩容 | 预估数据量 |

## 常见陷阱

1. **哈希函数质量**：分布不均会导致性能退化到 O(n)
2. **扩容时机**：太晚冲突多，太早浪费空间
3. **删除处理**：开放地址法删除要用墓碑标记，不能直接置空
4. **整数溢出**：哈希计算中使用无符号右移 `>>> 0`
