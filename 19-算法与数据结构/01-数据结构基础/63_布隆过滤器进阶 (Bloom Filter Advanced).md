# 64-布隆过滤器进阶 (Bloom Filter Advanced)

布隆过滤器的进阶应用和变种，包括计数布隆过滤器、缓存穿透防护等。

## 计数布隆过滤器（支持删除）

```javascript
class CountingBloomFilter {
  constructor(size = 1024, hashCount = 3) {
    this.counters = new Uint8Array(size);
    this.size = size;
    this.hashCount = hashCount;
  }

  _hash(item, seed) {
    let h = seed;
    for (const ch of String(item)) {
      h = (h * 31 + ch.charCodeAt(0)) % this.size;
    }
    return h;
  }

  add(item) {
    for (let i = 0; i < this.hashCount; i++) {
      this.counters[this._hash(item, i)]++;
    }
  }

  remove(item) {
    if (!this.has(item)) return false; // 防止误删
    for (let i = 0; i < this.hashCount; i++) {
      const idx = this._hash(item, i);
      if (this.counters[idx] > 0) this.counters[idx]--;
    }
    return true;
  }

  has(item) {
    for (let i = 0; i < this.hashCount; i++) {
      if (this.counters[this._hash(item, i)] === 0) return false;
    }
    return true;
  }
}
```

## C++ 实现

```cpp
#include <vector>
#include <string>
using namespace std;

class CountingBloomFilter {
    vector<int> counters;
    int sz;
    int k;

    int hash(const string& s, int seed) {
        unsigned h = seed;
        for (char c : s) h = h * 31 + c;
        return h % sz;
    }

public:
    CountingBloomFilter(int size = 1024, int hashCount = 3)
        : sz(size), k(hashCount), counters(size, 0) {}

    void add(const string& item) {
        for (int i = 0; i < k; i++) counters[hash(item, i)]++;
    }

    void remove(const string& item) {
        for (int i = 0; i < k; i++) {
            int idx = hash(item, i);
            if (counters[idx] > 0) counters[idx]--;
        }
    }

    bool has(const string& item) {
        for (int i = 0; i < k; i++) {
            if (counters[hash(item, i)] == 0) return false;
        }
        return true;
    }
};
```

## 缓存穿透防护

```javascript
class CacheWithBloomFilter {
  constructor(capacity = 1000) {
    this.cache = new Map();
    this.bloom = new BloomFilter(10000, 3);
  }

  get(key) {
    // 先查布隆过滤器
    if (!this.bloom.has(key)) {
      return null; // 一定不存在，避免穿透
    }
    // 可能存在，查缓存
    return this.cache.get(key) || null;
  }

  set(key, value) {
    this.bloom.add(key);
    this.cache.set(key, value);
    // 缓存淘汰逻辑...
    if (this.cache.size > this.capacity) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }
}
```

## 布隆过滤器合并

```javascript
// 合并两个相同参数的布隆过滤器
function mergeBloomFilters(bf1, bf2) {
  if (bf1.size !== bf2.size || bf1.hashCount !== bf2.hashCount) {
    throw new Error('参数不一致');
  }
  const merged = new BloomFilter(bf1.size, bf1.hashCount);
  for (let i = 0; i < bf1.size; i++) {
    if (bf1.bits[i] || bf2.bits[i]) {
      merged.bits[i] = 1;
    }
  }
  return merged;
}
```

## Cuckoo 过滤器（支持删除的替代方案）

```javascript
// Cuckoo 过滤器：支持删除，比计数布隆更省空间
class CuckooFilter {
  constructor(bucketCount = 1024, entriesPerBucket = 4) {
    this.buckets = Array.from({length: bucketCount},
      () => new Array(entriesPerBucket).fill(0));
    this.bucketCount = bucketCount;
    this.entriesPerBucket = entriesPerBucket;
  }

  _hash(item) {
    let h1 = 0, h2 = 0;
    for (const ch of String(item)) {
      h1 = (h1 * 31 + ch.charCodeAt(0)) % this.bucketCount;
      h2 = (h2 * 37 + ch.charCodeAt(0)) % this.bucketCount;
    }
    return [h1, h2]; // 两个候选桶
  }

  add(item) {
    const [h1, h2] = this._hash(item);
    const fingerprint = item.length * 31 + item.charCodeAt(0) || 1;
    // 尝试放入两个候选桶
    for (const idx of [h1, h2]) {
      for (let i = 0; i < this.entriesPerBucket; i++) {
        if (this.buckets[idx][i] === 0) {
          this.buckets[idx][i] = fingerprint;
          return true;
        }
      }
    }
    return false; // 需要踢出已有元素（此处简化）
  }

  has(item) {
    const [h1, h2] = this._hash(item);
    const fingerprint = item.length * 31 + item.charCodeAt(0) || 1;
    return this.buckets[h1].includes(fingerprint) ||
           this.buckets[h2].includes(fingerprint);
  }
}
```

## 应用场景扩展

- **Redis 缓存穿透**：Redis 4.0+ 内置布隆过滤器模块
- **HBase**：使用布隆过滤器减少磁盘读取
- **Chrome**：恶意 URL 检测
- **Bitcoin**：SPV 节点用布隆过滤器查询交易
- **Cassandra**：减少不必要 SSTable 读取

## 变种对比

| 变种 | 删除支持 | 空间效率 | 误判率 |
|------|---------|---------|--------|
| 标准布隆 | 否 | 极高 | 低 |
| 计数布隆 | 是 | 较低 | 略高 |
| Cuckoo | 是 | 高 | 低 |
| Scalable | 否 | 自适应 | 低 |

## 常见陷阱

1. **计数器溢出**：计数布隆过滤器的计数器有上限
2. **假阳性累积**：多个布隆过滤器合并会增加假阳性率
3. **删除前提**：只有确认元素存在才能安全删除
4. **参数一致**：合并操作要求两个过滤器参数相同
