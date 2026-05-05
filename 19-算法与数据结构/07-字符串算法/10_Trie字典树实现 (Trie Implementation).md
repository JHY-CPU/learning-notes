# 11 - Trie 字典树实现 (Trie Implementation)

  ## 完整 JavaScript 实现

  以下是一个功能完整的 Trie 实现，包含插入、查找、前缀检查、删除和遍历操作：


```

class TrieNode {
  constructor() {
    this.children = {};     // 子节点映射: char → TrieNode
    this.isEnd = false;     // 是否为一个完整单词的结尾
    this.count = 0;         // (可选) 通过该节点的单词数
    this.prefixCount = 0;   // (可选) 以该节点为前缀的单词数
  }
}

class Trie {
  constructor() {
    this.root = new TrieNode();
  }

  /**
   * 插入一个单词到 Trie 中
   * 时间复杂度: O(m), m为单词长度
   */
  insert(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children[ch]) {
        node.children[ch] = new TrieNode();
      }
      node = node.children[ch];
      node.prefixCount++;
    }
    node.isEnd = true;
    node.count++;
  }

  /**
   * 查找单词是否完整存在
   * 时间复杂度: O(m)
   */
  search(word) {
    const node = this._traverse(word);
    return node !== null && node.isEnd;
  }

  /**
   * 检查是否有单词以给定前缀开头
   * 时间复杂度: O(m)
   */
  startsWith(prefix) {
    return this._traverse(prefix) !== null;
  }

  /**
   * 查找单词出现的次数（允许重复插入）
   */
  count(word) {
    const node = this._traverse(word);
    return node ? node.count : 0;
  }

  /**
   * 获取以指定前缀开头的所有单词
   */
  getWordsWithPrefix(prefix) {
    const node = this._traverse(prefix);
    if (!node) return [];
    const result = [];
    this._dfs(node, prefix, result);
    return result;
  }

  /**
   * 删除一个单词
   * 如果该单词是其他单词的前缀，只取消 isEnd 标记
   * 如果没有其他单词共享前缀，则删除节点
   */
  delete(word) {
    const path = [];
    let node = this.root;
    for (const ch of word) {
      if (!node.children[ch]) return false; // 不存在
      path.push({ node, ch });
      node = node.children[ch];
    }
    if (!node.isEnd) return false; // 不存在

    node.isEnd = false;
    node.count--;

    // 从后向前清理没有子节点的路径
    for (let i = path.length - 1; i >= 0; i--) {
      const { node: parent, ch } = path[i];
      const child = parent.children[ch];
      child.prefixCount--;
      if (child.prefixCount === 0) {
        delete parent.children[ch];
      }
    }
    return true;
  }

  /**
   * 获取所有单词（字典序）
   */
  getAllWords() {
    const result = [];
    this._dfs(this.root, '', result);
    return result;
  }

  /**
   * 获取 Trie 中的单词总数
   */
  size() {
    return this.root.prefixCount;
  }

  // ========== 内部辅助方法 ==========

  _traverse(str) {
    let node = this.root;
    for (const ch of str) {
      if (!node.children[ch]) return null;
      node = node.children[ch];
    }
    return node;
  }

  _dfs(node, prefix, result) {
    if (node.isEnd) {
      for (let i = 0; i < node.count; i++) result.push(prefix);
    }
    // 按字典序遍历子节点
    const sortedKeys = Object.keys(node.children).sort();
    for (const ch of sortedKeys) {
      this._dfs(node.children[ch], prefix + ch, result);
    }
  }
}
  ```

  ## 核心操作详解

  ### 插入操作


```javascript

示例: 插入 "cat"

初始: root → {}
Step 1: root → {c: ...}    (创建节点 c)
Step 2: c → {a: ...}        (创建节点 a)
Step 3: a → {t: ...}        (创建节点 t)
Step 4: t.isEnd = true      (标记 t 为单词结尾)

最终:
  root → c → a → t(isEnd=true)
  ```

  ### 删除操作


```javascript

示例: 从 {"and", "ant"} 中删除 "and"

初始:
  root → a → n → d(isEnd=true)
               → t(isEnd=true)

删除 "and": 取消 d.isEnd
  检查 d 是否有子节点 → 没有，删除 d
  检查 n 是否有其他子节点 → 有(t) → 停止

最终:
  root → a → n → t(isEnd=true)
  ```

  ## 交互：完整 Trie 操作演示










  cat
car
card
care
dog
dot
do
