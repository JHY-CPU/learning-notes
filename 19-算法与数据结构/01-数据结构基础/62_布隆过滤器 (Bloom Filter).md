# 63-布隆过滤器 (Bloom Filter)

布隆过滤器是一种空间高效的概率型数据结构，用于判断元素是否存在于集合中。允许一定的假阳性误判，但绝无假阴性。

## 基本原理

1. 初始化一个长度为 m 的位数组，全部置为 0
2. 选择 k 个独立的哈希函数
3. 添加元素：用 k 个哈希函数计算位置，将对应位设为 1
4. 查询元素：检查 k 个位置是否全为 1

## JavaScript 实现

```javascript
class BloomFilter {
  constructor(size = 1024, hashCount = 3) {
    this.bits = new Uint8Array(size); // 位数组
    this.size = size;
    this.hashCount = hashCount;
  }

  _hash(item, seed) {
    let h = seed;
    const str = String(item);
    for (let i = 0; i < str.length; i++) {
      h = ((h * 31) + str.charCodeAt(i)) % this.size;
    }
    return h;
  }

  add(item) {
    for (let i = 0; i < this.hashCount; i++) {
      this.bits[this._hash(item, i)] = 1;
    }
  }

  has(item) {
    for (let i = 0; i < this.hashCount; i++) {
      if (this.bits[this._hash(item, i)] === 0) return false;
    }
    return true; // 可能存在（有假阳性）
  }
}

// 使用
const bf = new BloomFilter(1024, 3);
bf.add('apple');
bf.add('banana');
console.log(bf.has('apple'));  // true
console.log(bf.has('banana')); // true
console.log(bf.has('cherry')); // false（大概率）
```

## C++ 实现

```cpp
#include <vector>
#include <string>
#include <functional>
using namespace std;

class BloomFilter {
    vector<bool> bits;
    int size;

    int hash1(const string& s) {
        unsigned h = 5381;
        for (char c : s) h = ((h << 5) + h) + c;
        return h % size;
    }

    int hash2(const string& s) {
        unsigned h = 0;
        for (char c : s) h = h * 31 + c;
        return h % size;
    }

    int hash3(const string& s) {
        unsigned h = 2166136261u;
        for (char c : s) h = (h * 16777619) ^ c;
        return h % size;
    }

public:
    BloomFilter(int s = 1024) : size(s), bits(s) {}

    void add(const string& item) {
        bits[hash1(item)] = true;
        bits[hash2(item)] = true;
        bits[hash3(item)] = true;
    }

    bool has(const string& item) {
        return bits[hash1(item)] && bits[hash2(item)] && bits[hash3(item)];
    }
};
```

## 误判率计算

```
假阳性率 ≈ (1 - e^(-kn/m))^k

其中：
m = 位数组大小
n = 已插入元素数
k = 哈希函数数量

最优哈希函数数量：k = (m/n) * ln(2)

示例：m = 10000, n = 1000
k_opt = (10000/1000) * 0.693 ≈ 7
误判率 ≈ 0.8%
```

## 参数选择

| 元素数量 | 推荐位数组大小 (m) | 哈希函数数 (k) | 误判率 |
|---------|-------------------|---------------|--------|
| 1,000 | 9,585 | 7 | ~1% |
| 10,000 | 95,851 | 7 | ~1% |
| 100,000 | 958,506 | 7 | ~1% |
| 1,000,000 | 9,585,059 | 7 | ~1% |

公式：m = -n * ln(p) / (ln2)^2

## 布隆过滤器 vs 哈希表

| 特性 | 布隆过滤器 | 哈希表 |
|------|-----------|--------|
| 空间 | 极小 | 较大 |
| 查询 | O(k) | O(1) |
| 假阳性 | 有 | 无 |
| 假阴性 | 无 | 无 |
| 删除 | 不支持* | 支持 |
| 存储元素 | 不存储 | 存储 |

*计数布隆过滤器支持删除

## 应用场景

- **缓存穿透防护**：先查布隆过滤器，不存在则直接返回
- **垃圾邮件过滤**：快速判断邮件地址是否在黑名单中
- **爬虫 URL 去重**：避免重复爬取相同页面
- **推荐系统**：判断用户是否已看过某内容
- **分布式系统**：减少不必要的网络查询

## 常见陷阱

1. **假阳性**：布隆过滤器说"存在"不一定真的存在
2. **不支持删除**：删除一个元素可能影响其他元素的判断
3. **参数选择**：位数组太小或哈希函数太少会增加误判率
4. **不可变集合**：添加元素后无法恢复
