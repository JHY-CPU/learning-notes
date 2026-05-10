# 哈希专题 (Hash Problems)

## 一、概念定义与原理

### 1.1 哈希表

哈希表是一种通过**哈希函数**将键映射到数组下标的数据结构，实现 $O(1)$ 期望时间的查找。

### 1.2 哈希冲突

不同的键映射到同一个下标。解决方法：
- **链地址法：** 每个位置维护一个链表
- **开放地址法：** 冲突时探测下一个空位
- **双重哈希：** 用第二个哈希函数确定探测步长

### 1.3 良好哈希函数的要求

- 计算快速
- 分布均匀
- 冲突率低

---

## 二、核心算法

### 2.1 字符串哈希

将字符串映射为整数：

$$H(s) = \sum_{i=0}^{n-1} s[i] \cdot base^{n-1-i} \pmod{MOD}$$

常用参数：$base = 131$ 或 $13331$，$MOD = 2^{64}$（自然溢出）或 $10^9+7$

### 2.2 滚动哈希

支持 $O(1)$ 更新子串哈希值：

$$H(s[l+1 \ldots r+1]) = (H(s[l \ldots r]) - s[l] \cdot base^{r-l}) \cdot base + s[r+1]$$

### 2.3 一致性哈希

分布式系统中的负载均衡技术，将数据和服务器映射到一个环上。

---

## 三、代码实现

### 3.1 字符串哈希 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct StringHash {
    static const long long BASE = 131;
    static const long long MOD = 1e9 + 9;
    vector<long long> h, pw;

    void init(const string& s) {
        int n = s.size();
        h.resize(n + 1); pw.resize(n + 1);
        pw[0] = 1;
        for (int i = 0; i < n; i++) {
            h[i+1] = (h[i] * BASE + s[i]) % MOD;
            pw[i+1] = pw[i] * BASE % MOD;
        }
    }

    // 获取 s[l..r] 的哈希值
    long long get(int l, int r) {
        return (h[r+1] - h[l] * pw[r-l+1] % MOD + MOD) % MOD;
    }
};

// 比较 s[l1..r1] 和 s[l2..r2] 是否相等
bool equal_substring(StringHash& sh, int l1, int r1, int l2, int r2) {
    return sh.get(l1, r1) == sh.get(l2, r2);
}
```

### 3.2 双哈希（更安全）- C++

```cpp
struct DoubleHash {
    static const long long B1 = 131, B2 = 13331;
    static const long long M1 = 1e9+7, M2 = 1e9+9;
    vector<long long> h1, h2, pw1, pw2;

    void init(const string& s) {
        int n = s.size();
        h1.resize(n+1); h2.resize(n+1);
        pw1.resize(n+1); pw2.resize(n+1);
        pw1[0] = pw2[0] = 1;
        for (int i = 0; i < n; i++) {
            h1[i+1] = (h1[i]*B1 + s[i]) % M1;
            h2[i+1] = (h2[i]*B2 + s[i]) % M2;
            pw1[i+1] = pw1[i]*B1 % M1;
            pw2[i+1] = pw2[i]*B2 % M2;
        }
    }

    pair<long long,long long> get(int l, int r) {
        return {
            (h1[r+1] - h1[l]*pw1[r-l+1]%M1 + M1) % M1,
            (h2[r+1] - h2[l]*pw2[r-l+1]%M2 + M2) % M2
        };
    }
};
```

### 3.3 Python 实现

```python
class StringHash:
    BASE = 131
    MOD = 10**9 + 9

    def __init__(self, s):
        n = len(s)
        self.h = [0] * (n + 1)
        self.pw = [1] * (n + 1)
        for i, c in enumerate(s):
            self.h[i+1] = (self.h[i] * self.BASE + ord(c)) % self.MOD
            self.pw[i+1] = self.pw[i] * self.BASE % self.MOD

    def get(self, l, r):
        return (self.h[r+1] - self.h[l] * self.pw[r-l+1]) % self.MOD

sh = StringHash("abcabc")
print(sh.get(0, 2) == sh.get(3, 5))  # True, "abc" == "abc"
```

### 3.4 哈希表实现（链地址法）

```cpp
// 简单哈希表实现
const int MOD = 10007;
vector<pair<string,int>> table[MOD];

void insert(string key, int val) {
    int h = 0;
    for (char c : key) h = (h * 131 + c) % MOD;
    for (auto& [k, v] : table[h]) {
        if (k == key) { v = val; return; }
    }
    table[h].push_back({key, val});
}

int query(string key) {
    int h = 0;
    for (char c : key) h = (h * 131 + c) % MOD;
    for (auto& [k, v] : table[h]) {
        if (k == key) return v;
    }
    return -1;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度（期望） | 时间复杂度（最坏） |
|------|-----------------|-----------------|
| 插入 | $O(1)$ | $O(n)$ |
| 查找 | $O(1)$ | $O(n)$ |
| 删除 | $O(1)$ | $O(n)$ |
| 构建字符串哈希 | $O(n)$ | $O(n)$ |
| 子串哈希查询 | $O(1)$ | $O(1)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 1：** 两数之和（哈希查找）
2. **LeetCode 3：** 最长无重复子串（哈希+滑动窗口）
3. **字符串匹配：** Rabin-Karp 算法
4. **判重：** 用哈希判集合是否相等
5. **字符串哈希：** 快速比较子串是否相等
