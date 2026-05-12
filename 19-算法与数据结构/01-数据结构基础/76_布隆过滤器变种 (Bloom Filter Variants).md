# 77-布隆过滤器变种 (Bloom Filter Variants)

布隆过滤器有多种变种，针对不同场景进行了优化。

## 可扩展布隆过滤器

```javascript
class ScalableBloomFilter {
  constructor(initialSize = 100, growthFactor = 2, maxFilters = 5) {
    this.filters = [];
    this.growthFactor = growthFactor;
    this.maxFilters = maxFilters;
    this._addFilter(initialSize);
  }

  _addFilter(size) {
    this.filters.push({
      bits: new Uint8Array(size),
      size,
      count: 0
    });
  }

  _hash(item, seed, size) {
    let h = seed;
    for (const ch of String(item)) {
      h = (h * 31 + ch.charCodeAt(0)) % size;
    }
    return h;
  }

  add(item) {
    const f = this.filters[this.filters.length - 1];
    for (let i = 0; i < 3; i++) {
      f.bits[this._hash(item, i, f.size)] = 1;
    }
    f.count++;
    // 负载因子超过 70% 时创建新过滤器
    if (f.count > f.size * 0.7 && this.filters.length < this.maxFilters) {
      this._addFilter(f.size * this.growthFactor);
    }
  }

  has(item) {
    // 检查所有过滤器
    for (const f of this.filters) {
      let found = true;
      for (let i = 0; i < 3; i++) {
        if (f.bits[this._hash(item, i, f.size)] === 0) {
          found = false;
          break;
        }
      }
      if (found) return true;
    }
    return false;
  }
}
```

## C++ 实现

```cpp
#include <vector>
#include <string>
using namespace std;

struct BloomFilterLayer {
    vector<bool> bits;
    int size;
    int count;
    BloomFilterLayer(int s) : bits(s, false), size(s), count(0) {}
};

class ScalableBloomFilter {
    vector<BloomFilterLayer*> filters;
    int growthFactor;
    int maxFilters;

    int hash(const string& s, int seed, int size) {
        unsigned h = seed;
        for (char c : s) h = h * 31 + c;
        return h % size;
    }

public:
    ScalableBloomFilter(int initSize = 100, int gf = 2, int maxF = 5)
        : growthFactor(gf), maxFilters(maxF) {
        filters.push_back(new BloomFilterLayer(initSize));
    }

    void add(const string& item) {
        auto* f = filters.back();
        for (int i = 0; i < 3; i++) f->bits[hash(item, i, f->size)] = true;
        f->count++;
        if (f->count > f->size * 0.7 && (int)filters.size() < maxFilters) {
            filters.push_back(new BloomFilterLayer(f->size * growthFactor));
        }
    }

    bool has(const string& item) {
        for (auto* f : filters) {
            bool found = true;
            for (int i = 0; i < 3; i++)
                if (!f->bits[hash(item, i, f->size)]) { found = false; break; }
            if (found) return true;
        }
        return false;
    }
};
```

## 空间优化布隆过滤器

```javascript
// 使用位运算压缩存储
class CompactBloomFilter {
  constructor(size = 1024, hashCount = 3) {
    // 用 Int32Array 存储，每 32 位一个整数
    this.bits = new Int32Array(Math.ceil(size / 32));
    this.size = size;
    this.hashCount = hashCount;
  }

  _setBit(pos) {
    this.bits[pos >> 5] |= (1 << (pos & 31));
  }

  _getBit(pos) {
    return (this.bits[pos >> 5] >> (pos & 31)) & 1;
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
      this._setBit(this._hash(item, i));
    }
  }

  has(item) {
    for (let i = 0; i < this.hashCount; i++) {
      if (!this._getBit(this._hash(item, i))) return false;
    }
    return true;
  }
}
```

## 分层布隆过滤器

```javascript
// 用不同精度的多层过滤器减少误判
class LayeredBloomFilter {
  constructor() {
    // 第一层：快速粗筛（小容量）
    this.coarse = new BloomFilter(1000, 2);
    // 第二层：精确检查（大容量）
    this.fine = new BloomFilter(10000, 5);
  }

  add(item) {
    this.coarse.add(item);
    this.fine.add(item);
  }

  has(item) {
    // 先粗筛，再精确
    if (!this.coarse.has(item)) return false;
    return this.fine.has(item);
  }
}
```

## 变种对比

| 变种 | 删除 | 扩容 | 空间 | 误判率 |
|------|------|------|------|--------|
| 标准布隆 | 否 | 否 | 极小 | 低 |
| 计数布隆 | 是 | 否 | 4-8x | 略高 |
| 可扩展布隆 | 否 | 是 | 自适应 | 低 |
| Cuckoo 过滤器 | 是 | 否 | 较小 | 低 |
| 分层布隆 | 否 | 否 | 较大 | 极低 |
| 空间优化布隆 | 否 | 否 | 最小 | 低 |

## 选择指南

- **只需判断存在**：标准布隆过滤器
- **需要删除操作**：计数布隆过滤器或 Cuckoo 过滤器
- **数据量不确定**：可扩展布隆过滤器
- **内存极度受限**：空间优化布隆过滤器
- **需要极低误判率**：分层布隆过滤器
