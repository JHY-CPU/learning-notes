# 52-字符串哈希 (String Hashing)

字符串哈希（滚动哈希）将字符串映射为数值，实现 O(1) 子串比较，是 Rabin-Karp 算法的基础。

## 基础哈希函数

```javascript
// 多项式哈希
function strHash(s, base = 31, mod = 1e9 + 7) {
  let hash = 0, pow = 1;
  for (let i = 0; i < s.length; i++) {
    hash = (hash + (s.charCodeAt(i) - 96) * pow) % mod;
    pow = (pow * base) % mod;
  }
  return hash;
}
```

## 滚动哈希（前缀哈希）

```javascript
// 预处理 O(n)，查询 O(1)
class StringHasher {
  constructor(s, base = 31, mod = 1e9 + 7) {
    this.s = s;
    this.base = base;
    this.mod = mod;
    this.n = s.length;

    // 前缀哈希
    this.hash = new Array(n + 1).fill(0);
    this.pow = new Array(n + 1).fill(1);

    for (let i = 0; i < this.n; i++) {
      this.hash[i + 1] = (this.hash[i] * base + s.charCodeAt(i)) % mod;
      this.pow[i + 1] = (this.pow[i] * base) % mod;
    }
  }

  // 获取子串 [l, r] 的哈希值 O(1)
  getHash(l, r) {
    return ((this.hash[r + 1] - this.hash[l] * this.pow[r - l + 1]) % this.mod + this.mod) % this.mod;
  }
}
```

## C++ 实现

```cpp
#include <string>
#include <vector>
using namespace std;

class StringHasher {
    static const long long BASE = 31;
    static const long long MOD = 1e9 + 7;
    vector<long long> h, p;
    string s;
public:
    StringHasher(const string& str) : s(str) {
        int n = s.size();
        h.resize(n + 1, 0);
        p.resize(n + 1, 1);
        for (int i = 0; i < n; i++) {
            h[i + 1] = (h[i] * BASE + s[i]) % MOD;
            p[i + 1] = p[i] * BASE % MOD;
        }
    }

    long long getHash(int l, int r) {
        return ((h[r + 1] - h[l] * p[r - l + 1] % MOD) + MOD * 2) % MOD;
    }
};
```

## Rabin-Karp 字符串匹配

```javascript
function rabinKarp(text, pattern) {
  const hasher = new StringHasher(text);
  const pHasher = new StringHasher(pattern);
  const pHash = pHasher.getHash(0, pattern.length - 1);
  const res = [];

  for (let i = 0; i <= text.length - pattern.length; i++) {
    if (hasher.getHash(i, i + pattern.length - 1) === pHash) {
      // 哈希匹配，仍需逐字符确认（避免哈希冲突）
      let match = true;
      for (let j = 0; j < pattern.length; j++) {
        if (text[i + j] !== pattern[j]) { match = false; break; }
      }
      if (match) res.push(i);
    }
  }
  return res;
}

console.log(rabinKarp("ababcabcabababd", "ababd")); // [10]
```

## 双哈希降低冲突概率

```javascript
class DoubleHasher {
  constructor(s) {
    this.s = s;
    this.mod1 = 1e9 + 7;
    this.mod2 = 1e9 + 9;
    this.base1 = 31;
    this.base2 = 37;
    this.n = s.length;
    this._init();
  }

  _init() {
    this.h1 = new Array(this.n + 1).fill(0);
    this.h2 = new Array(this.n + 1).fill(0);
    this.p1 = new Array(this.n + 1).fill(1);
    this.p2 = new Array(this.n + 1).fill(1);
    for (let i = 0; i < this.n; i++) {
      const c = this.s.charCodeAt(i);
      this.h1[i+1] = (this.h1[i] * this.base1 + c) % this.mod1;
      this.h2[i+1] = (this.h2[i] * this.base2 + c) % this.mod2;
      this.p1[i+1] = (this.p1[i] * this.base1) % this.mod1;
      this.p2[i+1] = (this.p2[i] * this.base2) % this.mod2;
    }
  }

  getHash(l, r) {
    const v1 = ((this.h1[r+1] - this.h1[l] * this.p1[r-l+1]) % this.mod1 + this.mod1) % this.mod1;
    const v2 = ((this.h2[r+1] - this.h2[l] * this.p2[r-l+1]) % this.mod2 + this.mod2) % this.mod2;
    return [v1, v2];
  }

  isEqual(l1, r1, l2, r2) {
    const h1 = this.getHash(l1, r1);
    const h2 = this.getHash(l2, r2);
    return h1[0] === h2[0] && h1[1] === h2[1];
  }
}
```

## 应用

- **字符串匹配**：Rabin-Karp O(n + m) 匹配
- **最长重复子串**：二分长度 + 哈希检查
- **最长公共子串**：双哈希 + 二分
- **回文判断**：正序哈希 == 逆序哈希
- **子串去重**：哈希值集合去重

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 预处理 | O(n) | O(n) |
| 子串哈希查询 | O(1) | - |
| Rabin-Karp 匹配 | O(n + m)* | O(1) |
| 双哈希查询 | O(1) | O(n) |

*平均情况，最坏 O(nm)（哈希冲突）

## 常见陷阱

1. **取模溢出**：计算中要注意防止溢出
2. **负数取模**：`(a % mod + mod) % mod` 保证非负
3. **哈希冲突**：关键场景使用双哈希
4. **base 选择**：选择质数作为 base，避免冲突
