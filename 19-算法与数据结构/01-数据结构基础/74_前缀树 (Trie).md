# 75-前缀树 (Trie)

前缀树（字典树）利用字符串的公共前缀减少查询次数，实现 O(len) 的前缀匹配和搜索。

## 基本实现

```javascript
class TrieNode {
  constructor() {
    this.children = {};
    this.isEnd = false;
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
  }

  // 插入单词 O(len)
  insert(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children[ch]) node.children[ch] = new TrieNode();
      node = node.children[ch];
    }
    node.isEnd = true;
  }

  // 精确查找单词 O(len)
  search(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children[ch]) return false;
      node = node.children[ch];
    }
    return node.isEnd;
  }

  // 前缀查找 O(len)
  startsWith(prefix) {
    let node = this.root;
    for (const ch of prefix) {
      if (!node.children[ch]) return false;
      node = node.children[ch];
    }
    return true;
  }

  // 删除单词
  delete(word) {
    const _delete = (node, depth) => {
      if (depth === word.length) {
        node.isEnd = false;
        return Object.keys(node.children).length === 0;
      }
      const ch = word[depth];
      if (!node.children[ch]) return false;
      const shouldDelete = _delete(node.children[ch], depth + 1);
      if (shouldDelete) {
        delete node.children[ch];
        return !node.isEnd && Object.keys(node.children).length === 0;
      }
      return false;
    };
    _delete(this.root, 0);
  }

  // 获取所有以 prefix 开头的单词
  autocomplete(prefix) {
    let node = this.root;
    for (const ch of prefix) {
      if (!node.children[ch]) return [];
      node = node.children[ch];
    }
    const results = [];
    const dfs = (curr, path) => {
      if (curr.isEnd) results.push(path);
      for (const [ch, child] of Object.entries(curr.children)) {
        dfs(child, path + ch);
      }
    };
    dfs(node, prefix);
    return results;
  }
}
```

## C++ 实现

```cpp
#include <string>
#include <vector>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    bool isEnd;
    TrieNode() : isEnd(false) {
        for (int i = 0; i < 26; i++) children[i] = nullptr;
    }
};

class Trie {
    TrieNode* root;
public:
    Trie() : root(new TrieNode()) {}

    void insert(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            int idx = ch - 'a';
            if (!node->children[idx]) node->children[idx] = new TrieNode();
            node = node->children[idx];
        }
        node->isEnd = true;
    }

    bool search(const string& word) {
        TrieNode* node = root;
        for (char ch : word) {
            int idx = ch - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return node->isEnd;
    }

    bool startsWith(const string& prefix) {
        TrieNode* node = root;
        for (char ch : prefix) {
            int idx = ch - 'a';
            if (!node->children[idx]) return false;
            node = node->children[idx];
        }
        return true;
    }
};
```

## 典型应用

```javascript
// 1. 单词搜索 II（在矩阵中找所有单词）
function findWords(board, words) {
  const trie = new Trie();
  for (const w of words) trie.insert(w);

  const result = new Set();
  const m = board.length, n = board[0].length;

  function dfs(i, j, node, path) {
    if (i < 0 || i >= m || j < 0 || j >= n) return;
    const ch = board[i][j];
    if (!node.children[ch]) return;

    node = node.children[ch];
    path += ch;
    if (node.isEnd) result.add(path);

    board[i][j] = '#'; // 标记已访问
    dfs(i+1, j, node, path);
    dfs(i-1, j, node, path);
    dfs(i, j+1, node, path);
    dfs(i, j-1, node, path);
    board[i][j] = ch; // 恢复
  }

  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      dfs(i, j, trie.root, '');

  return [...result];
}

// 2. 实现字典（拼写检查）
class SpellChecker {
  constructor(words) {
    this.trie = new Trie();
    for (const w of words) this.trie.insert(w.toLowerCase());
  }
  check(word) { return this.trie.search(word.toLowerCase()); }
  suggest(prefix) { return this.trie.autocomplete(prefix.toLowerCase()); }
}
```

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| insert | O(len) | O(len * 26) 最坏 |
| search | O(len) | O(1) |
| startsWith | O(len) | O(1) |
| delete | O(len) | O(1) |

## Trie vs 哈希表

| 特性 | Trie | 哈希表 |
|------|------|--------|
| 前缀匹配 | O(len) 高效 | 需遍历 |
| 精确匹配 | O(len) | O(1) |
| 空间 | 共享前缀省空间 | 每键独立 |
| 排序输出 | 天然有序 | 需额外排序 |

## 应用场景

- 搜索引擎自动补全
- 拼写检查
- IP 路由最长前缀匹配
- 文本过滤和敏感词检测
- 词频统计和文本分析

## 常见陷阱

1. **内存消耗**：每个节点有 26 个指针，小字母表更高效
2. **delete 不清理空节点**：要递归清理不再使用的节点
3. **Unicode 支持**：处理中文等多字节字符需要更复杂的节点结构
4. **大写/小写**：根据需求统一大小写
