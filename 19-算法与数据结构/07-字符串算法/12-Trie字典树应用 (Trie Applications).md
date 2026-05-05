# 12 - Trie 字典树应用 (Trie Applications)

  ## 应用场景概览

  Trie 结构在以下场景中有广泛应用：


    - **自动补全/输入提示** — 搜索引擎、IDE、手机输入法

    - **拼写检查** — 词库匹配与纠错建议

    - **IP 路由（最长前缀匹配）** — 路由器查找下一跳

    - **词频统计** — 在节点中记录 count

    - **字符串排序** — Trie 的先序遍历即是字典序

    - **敏感词过滤** — 结合 AC 自动机实现

    - **电话号码簿** — 快速查找和前缀匹配



  ## 应用1：自动补全（AutoComplete）

  自动补全是 Trie 最经典的应用。用户在输入框中键入前缀，系统实时给出以该前缀开头的所有单词建议。


```

/**
 * Trie 自动补全功能
 */
class Trie {
  // ... 基本操作同上 ...

  /**
   * 获取以 prefix 为前缀的所有单词
   * 用于自动补全建议
   */
  autoComplete(prefix, limit = 10) {
    const node = this._traverse(prefix);
    if (!node) return [];

    const suggestions = [];
    this._dfsWithLimit(node, prefix, suggestions, limit);
    return suggestions;
  }

  _dfsWithLimit(node, prefix, result, limit) {
    if (result.length >= limit) return;
    if (node.isEnd) result.push(prefix);
    const keys = Object.keys(node.children).sort();
    for (const ch of keys) {
      if (result.length >= limit) break;
      this._dfsWithLimit(node.children[ch], prefix + ch, result, limit);
    }
  }
}
  ```

  ## 应用2：最长公共前缀（Longest Common Prefix）

  在 Trie 中，从根节点到第一个分叉点之间的路径就是所有单词的最长公共前缀。


```javascript

// 在一组字符串中找最长公共前缀
function longestCommonPrefix(words) {
  if (!words || words.length === 0) return '';
  if (words.length === 1) return words[0];

  // 构建 Trie
  const trie = new Trie();
  for (const w of words) trie.insert(w);

  // 从根节点开始，只要只有一个子节点就继续
  let node = trie.root;
  let prefix = '';
  while (node && !node.isEnd && Object.keys(node.children).length === 1) {
    const ch = Object.keys(node.children)[0];
    prefix += ch;
    node = node.children[ch];
  }
  return prefix;
}

// 示例: ["flower", "flow", "flight"] → "fl"
  ```

  ## 应用3：词频统计与 Top-K

  在 Trie 节点中记录 count 字段，可以统计单词出现频率，进而找出频率最高的 K 个单词：


```

function topKFrequent(words, k) {
  const trie = new Trie();
  for (const w of words) trie.insert(w);

  // 遍历所有单词并统计频率
  const freq = {};
  function dfs(node, prefix) {
    if (node.isEnd) freq[prefix] = node.count;
    for (const [ch, child] of Object.entries(node.children)) {
      dfs(child, prefix + ch);
    }
  }
  dfs(trie.root, '');

  // 排序取前 K 个
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, k)
    .map(entry => entry[0]);
}
  ```

  ## 应用4：单词搜索（DFS + Trie）

  LeetCode 212. 单词搜索 II — 在二维字符网格中找出所有由相邻单元格组成的单词：


```javascript

function findWords(board, words) {
  const trie = new Trie();
  for (const w of words) trie.insert(w);

  const result = new Set();
  const m = board.length, n = board[0].length;

  function dfs(i, j, node, path) {
    if (node.isEnd) result.add(path);
    if (i < 0 || i >= m || j < 0 || j >= n) return;
    const ch = board[i][j];
    if (ch === '#' || !node.children[ch]) return;

    board[i][j] = '#'; // 标记已访问
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      dfs(i + dx, j + dy, node.children[ch], path + ch);
    }
    board[i][j] = ch; // 恢复
  }

  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      dfs(i, j, trie.root, '');

  return [...result];
}
  ```

  ## 交互演示：自动补全

  在下方输入前缀，实时查看自动补全建议：

  apple
application
apply
apprentice
banana
band
banner
cat
catalog
catch
category
