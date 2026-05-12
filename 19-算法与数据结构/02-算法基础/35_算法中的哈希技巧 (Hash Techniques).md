# 36-算法中的哈希技巧 (Hash Techniques)

哈希将任意大小的数据映射到固定大小的值，在算法中用于快速查找、去重、字符串匹配等。

## 核心技巧

| 技巧 | 用途 | 复杂度 |
|------|------|--------|
| 字符串哈希 | O(1) 比较字符串 | O(n) 预处理 |
| 滚动哈希 | 增量更新哈希值 | O(1) 更新 |
| 双哈希 | 降低冲突概率 | 2x 空间 |
| 一致性哈希 | 分布式负载均衡 | O(log n) |

## 滚动哈希原理

对于字符串 s，多项式哈希：H(s) = (s[0]*p^(n-1) + s[1]*p^(n-2) + ... + s[n-1]) mod q
- 添加字符：H' = (H * p + c) mod q
- 移除首字符：H' = (H - s[0]*p^(n-1)) * p + new_c mod q

## JavaScript 实现

```javascript
// 字符串哈希（Rabin-Karp 用）
class StringHash {
  constructor(s, base = 31, mod = 1e9 + 7) {
    this.mod = mod;
    this.base = base;
    this.n = s.length;
    this.hash = new Array(n + 1).fill(0);
    this.pow = new Array(n + 1).fill(1);

    for (let i = 0; i < s.length; i++) {
      this.hash[i + 1] = (this.hash[i] * base + s.charCodeAt(i)) % mod;
      this.pow[i + 1] = (this.pow[i] * base) % mod;
    }
  }

  // 获取 s[l..r] 的哈希值
  getHash(l, r) {
    return (this.hash[r + 1] - this.hash[l] * this.pow[r - l + 1] % this.mod + this.mod) % this.mod;
  }
}

// Rabin-Karp 字符串匹配
function rabinKarp(text, pattern) {
  const n = text.length, m = pattern.length;
  const base = 31, mod = 1e9 + 7;

  let patternHash = 0;
  for (const c of pattern) patternHash = (patternHash * base + c.charCodeAt(0)) % mod;

  let textHash = 0;
  for (let i = 0; i < m; i++) textHash = (textHash * base + text.charCodeAt(i)) % mod;

  const basePow = Math.pow(base, m) % mod;
  const result = [];

  if (textHash === patternHash) result.push(0);

  for (let i = 1; i <= n - m; i++) {
    textHash = (textHash * base - text.charCodeAt(i - 1) * basePow % mod + mod) % mod;
    textHash = (textHash + text.charCodeAt(i + m - 1)) % mod;
    if (textHash === patternHash) result.push(i);
  }
  return result;
}

// 一致性哈希（简化版）
class ConsistentHash {
  constructor(nodes, replicas = 3) {
    this.replicas = replicas;
    this.ring = new Map();
    this.sortedKeys = [];
    for (const node of nodes) this.addNode(node);
  }

  addNode(node) {
    for (let i = 0; i < this.replicas; i++) {
      const hash = this._hash(`${node}:${i}`);
      this.ring.set(hash, node);
      this.sortedKeys.push(hash);
    }
    this.sortedKeys.sort((a, b) => a - b);
  }

  _hash(s) {
    let h = 0;
    for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 100000;
    return h;
  }

  get(key) {
    if (!this.sortedKeys.length) return null;
    const hash = this._hash(key);
    for (const k of this.sortedKeys) {
      if (k >= hash) return this.ring.get(k);
    }
    return this.ring.get(this.sortedKeys[0]);
  }
}

// 测试
console.log(rabinKarp('ABABDABACDABABCABAB', 'ABABCABAB')); // [9]
const ch = new ConsistentHash(['node1', 'node2', 'node3']);
console.log(ch.get('user123'));
```

## C++ 实现

```cpp
#include <string>
#include <vector>
#include <map>
using namespace std;

// 双哈希降低冲突概率
typedef long long ll;
const ll MOD1 = 1e9 + 7, MOD2 = 1e9 + 9;
const ll BASE = 31;

struct DoubleHash {
    vector<ll> h1, h2, p1, p2;
    DoubleHash(const string& s) {
        int n = s.size();
        h1.resize(n + 1); h2.resize(n + 1);
        p1.resize(n + 1); p2.resize(n + 1);
        p1[0] = p2[0] = 1;
        for (int i = 0; i < n; i++) {
            h1[i + 1] = (h1[i] * BASE + s[i]) % MOD1;
            h2[i + 1] = (h2[i] * BASE + s[i]) % MOD2;
            p1[i + 1] = p1[i] * BASE % MOD1;
            p2[i + 1] = p2[i] * BASE % MOD2;
        }
    }
    pair<ll, ll> getHash(int l, int r) {
        ll v1 = (h1[r + 1] - h1[l] * p1[r - l + 1] % MOD1 + MOD1) % MOD1;
        ll v2 = (h2[r + 1] - h2[l] * p2[r - l + 1] % MOD2 + MOD2) % MOD2;
        return {v1, v2};
    }
};

// Rabin-Karp
vector<int> rabinKarp(const string& text, const string& pattern) {
    DoubleHash ph(pattern), th(text);
    auto target = ph.getHash(0, pattern.size() - 1);
    vector<int> result;
    int m = pattern.size();
    for (int i = 0; i + m <= text.size(); i++) {
        if (th.getHash(i, i + m - 1) == target) result.push_back(i);
    }
    return result;
}
```

## 复杂度

| 算法 | 预处理 | 查询/匹配 |
|------|--------|----------|
| 字符串哈希 | O(n) | O(1) |
| Rabin-Karp | O(n + m) | O(n + m) |
| 一致性哈希 | O(n log n) | O(log n) |

## 常见陷阱

1. **模数选择**：必须用质数，否则冲突概率大
2. **基数选择**：应大于字符集大小（如 31 对小写字母）
3. **双哈希**：两个模数应互质
4. **溢出**：JavaScript 需用 BigInt 处理大数

## 实际应用

Rabin-Karp 字符串匹配算法使用滚动哈希在 O(n) 时间内找到所有匹配位置。一致性哈希广泛用于分布式缓存系统（如 Memcached、Redis Cluster）。哈希技巧在竞赛中常用于字符串匹配和去重。
