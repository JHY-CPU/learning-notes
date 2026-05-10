# 字符串匹配专题 (String Matching)

## 一、概念定义与原理

### 1.1 问题定义

给定文本 $T$（长度 $n$）和模式 $P$（长度 $m$），找出 $P$ 在 $T$ 中的所有出现位置。

### 1.2 算法分类

| 算法 | 时间复杂度 | 特点 |
|------|-----------|------|
| 朴素匹配 | $O(nm)$ | 简单但慢 |
| KMP | $O(n+m)$ | 利用前缀函数 |
| Rabin-Karp | $O(n+m)$ 期望 | 哈希比较 |
| Trie | $O(\sum|P_i|)$ 构建 | 多模式匹配 |
| AC自动机 | $O(\sum|P_i| + n + \text{匹配数})$ | 多模式匹配 |

---

## 二、核心算法

### 2.1 KMP 算法

**前缀函数 $\pi[i]$：** $P[0 \ldots i]$ 的最长真前缀同时也是后缀的长度。

**匹配过程：** 当 $T[i] \neq P[j]$ 时，$j = \pi[j-1]$（而不是回退到 0）。

### 2.2 Trie（字典树）

将所有模式串插入一棵树中，每个节点代表一个字符。查询时沿树走。

### 2.3 AC 自动机

Trie + 失配指针（类似 KMP 的 next 数组），支持多模式串同时匹配。

---

## 三、代码实现

### 3.1 KMP - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> build_pi(string p) {
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

vector<int> kmp(string t, string p) {
    auto pi = build_pi(p);
    vector<int> res;
    int j = 0;
    for (int i = 0; i < t.size(); i++) {
        while (j > 0 && t[i] != p[j]) j = pi[j-1];
        if (t[i] == p[j]) j++;
        if (j == p.size()) { res.push_back(i - j + 1); j = pi[j-1]; }
    }
    return res;
}
```

### 3.2 Trie - C++

```cpp
class Trie {
    struct Node { int children[26] = {}; int count = 0; bool end = false; };
    vector<Node> nodes;
public:
    Trie() { nodes.emplace_back(); }
    void insert(string s) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (!nodes[cur].children[idx]) {
                nodes[cur].children[idx] = nodes.size();
                nodes.emplace_back();
            }
            cur = nodes[cur].children[idx];
            nodes[cur].count++;
        }
        nodes[cur].end = true;
    }
    bool search(string s) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (!nodes[cur].children[idx]) return false;
            cur = nodes[cur].children[idx];
        }
        return nodes[cur].end;
    }
    bool starts_with(string prefix) {
        int cur = 0;
        for (char c : prefix) {
            int idx = c - 'a';
            if (!nodes[cur].children[idx]) return false;
            cur = nodes[cur].children[idx];
        }
        return true;
    }
};
```

### 3.3 AC 自动机 - C++

```cpp
class AhoCorasick {
    struct Node { int ch[26], fail, cnt; };
    vector<Node> nodes;
public:
    AhoCorasick() { nodes.push_back({{}, 0, 0}); }
    void insert(string s) {
        int cur = 0;
        for (char c : s) {
            int idx = c - 'a';
            if (!nodes[cur].ch[idx]) {
                nodes[cur].ch[idx] = nodes.size();
                nodes.push_back({{}, 0, 0});
            }
            cur = nodes[cur].ch[idx];
        }
        nodes[cur].cnt++;
    }
    void build() {
        queue<int> q;
        for (int i = 0; i < 26; i++)
            if (nodes[0].ch[i]) q.push(nodes[0].ch[i]);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int i = 0; i < 26; i++) {
                if (nodes[u].ch[i]) {
                    nodes[nodes[u].ch[i]].fail = nodes[nodes[u].fail].ch[i];
                    q.push(nodes[u].ch[i]);
                } else {
                    nodes[u].ch[i] = nodes[nodes[u].fail].ch[i];
                }
            }
        }
    }
    int query(string text) {
        int cur = 0, result = 0;
        for (char c : text) {
            cur = nodes[cur].ch[c - 'a'];
            for (int t = cur; t && nodes[t].cnt != -1; t = nodes[t].fail) {
                result += nodes[t].cnt;
                nodes[t].cnt = -1; // 标记已统计
            }
        }
        return result;
    }
};
```

### 3.4 Python 实现

```python
class Trie:
    def __init__(self):
        self.children = {}; self.end = False
    def insert(self, s):
        node = self
        for c in s:
            if c not in node.children: node.children[c] = Trie()
            node = node.children[c]
        node.end = True
    def search(self, s):
        node = self
        for c in s:
            if c not in node.children: return False
            node = node.children[c]
        return node.end

trie = Trie()
trie.insert("apple"); trie.insert("app")
print(trie.search("apple"))  # True
print(trie.search("app"))    # True
print(trie.search("ap"))     # False
```

---

## 四、复杂度分析

| 算法 | 构建 | 匹配 | 空间 |
|------|------|------|------|
| KMP | $O(m)$ | $O(n)$ | $O(m)$ |
| Trie | $O(\sum|P_i|)$ | $O(|s|)$ | $O(\sum|P_i| \cdot |\Sigma|)$ |
| AC自动机 | $O(\sum|P_i| \cdot |\Sigma|)$ | $O(n + \text{匹配数})$ | $O(\sum|P_i| \cdot |\Sigma|)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 28：** 找字符串第一个匹配位置
2. **LeetCode 208：** 实现Trie
3. **LeetCode 211：** 添加与搜索单词
4. **多关键词过滤：** AC 自动机
5. **文本编辑器搜索：** KMP
