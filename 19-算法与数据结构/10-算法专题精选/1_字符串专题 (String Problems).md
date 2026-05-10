# 字符串专题 (String Problems)

## 一、概念定义与原理

### 1.1 字符串匹配

给定文本 $T$ 和模式 $P$，找出 $P$ 在 $T$ 中的所有出现位置。

朴素算法 $O(nm)$，KMP 算法 $O(n+m)$。

### 1.2 KMP 算法

利用**前缀函数**（next 数组）避免回溯。$next[i]$ 表示 $P[0 \ldots i]$ 的最长真前缀同时也是后缀的长度。

### 1.3 Rabin-Karp

将字符串视为哈希值，通过滚动哈希快速比较。期望 $O(n+m)$，最坏 $O(nm)$。

### 1.4 后缀数组

将所有后缀排序后得到的数组。配合 LCP（最长公共前缀）数组，可以高效解决大量字符串问题。

---

## 二、核心算法

### 2.1 前缀函数（KMP 的核心）

$$\pi[i] = \max\{k \mid k \leq i, P[0 \ldots k-1] = P[i-k+1 \ldots i]\}$$

递推：若 $P[\pi[i-1]] = P[i]$，则 $\pi[i] = \pi[i-1] + 1$；否则递归回退。

### 2.2 滚动哈希

$$H(s) = \sum_{i=0}^{n-1} s[i] \cdot base^{n-1-i} \pmod{MOD}$$

滚动更新：$H(s[l+1 \ldots r+1]) = (H(s[l \ldots r]) - s[l] \cdot base^{r-l}) \cdot base + s[r+1]$

### 2.3 后缀数组构建

倍增法 $O(n \log n)$ 或 SA-IS $O(n)$。

---

## 三、代码实现

### 3.1 KMP 算法 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 构建前缀函数
vector<int> build_prefix(string p) {
    int m = p.size();
    vector<int> pi(m, 0);
    for (int i = 1; i < m; i++) {
        int j = pi[i-1];
        while (j > 0 && p[i] != p[j]) j = pi[j-1];
        if (p[i] == p[j]) j++;
        pi[i] = j;
    }
    return pi;
}

// KMP 匹配，返回所有匹配位置
vector<int> kmp(string text, string pattern) {
    vector<int> pi = build_prefix(pattern);
    vector<int> result;
    int n = text.size(), m = pattern.size();
    int j = 0;
    for (int i = 0; i < n; i++) {
        while (j > 0 && text[i] != pattern[j]) j = pi[j-1];
        if (text[i] == pattern[j]) j++;
        if (j == m) {
            result.push_back(i - m + 1);
            j = pi[j-1];
        }
    }
    return result;
}
```

### 3.2 Rabin-Karp - C++

```cpp
vector<int> rabin_karp(string text, string pattern) {
    const long long BASE = 131, MOD = 1e9 + 7;
    int n = text.size(), m = pattern.size();
    if (m > n) return {};

    // 计算 pattern 的哈希和 base^(m-1)
    long long p_hash = 0, t_hash = 0, power = 1;
    for (int i = 0; i < m; i++) {
        p_hash = (p_hash * BASE + pattern[i]) % MOD;
        t_hash = (t_hash * BASE + text[i]) % MOD;
        if (i > 0) power = power * BASE % MOD;
    }

    vector<int> result;
    for (int i = 0; i <= n - m; i++) {
        if (t_hash == p_hash) {
            if (text.substr(i, m) == pattern) result.push_back(i);
        }
        if (i < n - m) {
            t_hash = ((t_hash - text[i] * power) * BASE + text[i+m]) % MOD;
            if (t_hash < 0) t_hash += MOD;
        }
    }
    return result;
}
```

### 3.3 Python 实现

```python
def build_prefix(p):
    pi = [0] * len(p)
    for i in range(1, len(p)):
        j = pi[i-1]
        while j > 0 and p[i] != p[j]: j = pi[j-1]
        if p[i] == p[j]: j += 1
        pi[i] = j
    return pi

def kmp(text, pattern):
    pi = build_prefix(pattern)
    result = []; j = 0
    for i, c in enumerate(text):
        while j > 0 and c != pattern[j]: j = pi[j-1]
        if c == pattern[j]: j += 1
        if j == len(pattern):
            result.append(i - j + 1)
            j = pi[j-1]
    return result

print(kmp("ababcababc", "abc"))  # [2, 7]
print(build_prefix("aabaaab"))   # [0, 1, 0, 1, 2, 2, 3]
```

### 3.4 最长回文子串（Manacher）

```cpp
// Manacher 算法 O(n)
string manacher(string s) {
    string t = "#";
    for (char c : s) { t += c; t += "#"; }
    int n = t.size(), center = 0, right = 0;
    vector<int> p(n, 0);
    for (int i = 0; i < n; i++) {
        if (i < right) p[i] = min(right - i, p[2*center - i]);
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n &&
               t[i - p[i] - 1] == t[i + p[i] + 1]) p[i]++;
        if (i + p[i] > right) { center = i; right = i + p[i]; }
    }
    int max_len = 0, max_center = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] > max_len) { max_len = p[i]; max_center = i; }
    }
    int start = (max_center - max_len) / 2;
    return s.substr(start, max_len);
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| KMP | $O(n + m)$ | $O(m)$ |
| Rabin-Karp | $O(n + m)$ 期望 | $O(1)$ |
| Manacher | $O(n)$ | $O(n)$ |
| 后缀数组（倍增） | $O(n \log n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 28：** 找字符串第一个匹配位置（KMP）
2. **LeetCode 5：** 最长回文子串（Manacher）
3. **最长重复子串：** 后缀数组 + LCP
4. **字符串哈希：** 快速判断子串相等
