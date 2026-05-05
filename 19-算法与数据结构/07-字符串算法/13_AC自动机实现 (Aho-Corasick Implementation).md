# 14 - AC 自动机实现 (Aho-Corasick Implementation)

  ## 完整 JavaScript 实现


```

/**
 * AC 自动机节点
 */
class ACNode {
  constructor() {
    this.children = {};       // 子节点: char → ACNode
    this.fail = null;         // 失败指针
    this.output = [];         // 以该节点结尾的模式串列表
    this.depth = 0;           // 节点深度（根节点深度为 0）
  }
}

/**
 * Aho-Corasick 自动机
 */
class AhoCorasick {
  constructor() {
    this.root = new ACNode();
    this.built = false;       // 标记是否已构建失败指针
  }

  /**
   * 插入模式串到 Trie 树
   */
  insert(pattern) {
    if (this.built) {
      throw new Error('Cannot insert after build()');
    }
    let node = this.root;
    for (const ch of pattern) {
      if (!node.children[ch]) {
        node.children[ch] = new ACNode();
      }
      node = node.children[ch];
    }
    node.output.push(pattern);
  }

  /**
   * 构建失败指针（BFS）
   * 这是 AC 自动机的核心
   */
  build() {
    const queue = [];

    // 第 1 步：根节点的所有直接子节点的 fail 指向 root
    for (const ch in this.root.children) {
      const child = this.root.children[ch];
      child.fail = this.root;
      queue.push(child);
    }

    // 第 2 步：BFS 遍历，计算每个节点的 fail 指针
    while (queue.length > 0) {
      const node = queue.shift();

      for (const ch in node.children) {
        const child = node.children[ch];

        // 沿着父节点的 fail 链查找
        let f = node.fail;
        while (f && !f.children[ch]) {
          f = f.fail;
        }

        if (f) {
          child.fail = f.children[ch];
        } else {
          child.fail = this.root;
        }

        // ★ 关键优化：合并输出
        // 当前节点的输出 = 自身的输出 + fail 节点的输出
        child.output = child.output.concat(child.fail.output);

        queue.push(child);
      }
    }

    this.built = true;
  }

  /**
   * 在文本中搜索所有模式串
   * @param {string} text - 文本串
   * @returns {Array} [{pattern, position}, ...]
   */
  search(text) {
    if (!this.built) this.build();

    const results = [];
    let node = this.root;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];

      // 当前字符不匹配时，沿 fail 指针跳转
      while (node !== this.root && !node.children[ch]) {
        node = node.fail;
      }

      // 如果存在匹配的子节点，进入该节点
      if (node.children[ch]) {
        node = node.children[ch];
      }

      // 输出所有匹配结果
      for (const pattern of node.output) {
        results.push({
          pattern: pattern,
          position: i - pattern.length + 1
        });
      }
    }

    return results;
  }

  /**
   * 敏感词过滤（将匹配到的模式串替换为 ***）
   */
  filter(text, replaceChar = '*') {
    const matches = this.search(text);
    if (matches.length === 0) return text;

    let result = text.split('');
    for (const { pattern, position } of matches) {
      for (let i = position; i < position + pattern.length; i++) {
        result[i] = replaceChar;
      }
    }
    return result.join('');
  }
}
  ```

  ## 优化技术



    ****
    ****
    ****
    ****
  | 优化技术 | 描述 | 效果 |
| --- | --- | --- |
| 输出合并 | 构建失败指针时，将 fail 节点的输出合并到当前节点 | 匹配时无需再沿 fail 链查找输出 |
| 转移表优化 | 将不存在的子节点指向 fail.children，形成确定性自动机（DFA） | 匹配时无需 while 循环，每次 O(1) |
| 字典树压缩 | 对只有单一路径的节点进行压缩 | 减少节点数，节省空间 |
| 双数组 Trie | 用两个数组（base, check）实现 Trie 节点 | 大幅减少内存占用 |

  ### 转移表优化版


```

// 将 AC 自动机转化为 DFA（确定性有限自动机）
// 这样匹配时不需要 while 循环
toDFA() {
  const queue = [this.root];
  while (queue.length > 0) {
    const node = queue.shift();
    for (const ch in node.children) {
      queue.push(node.children[ch]);
    }
    // 为每个可能字符补充转移
    for (let c = 0; c < 256; c++) {
      const ch = String.fromCharCode(c);
      if (!node.children[ch]) {
        // 如果当前节点没有对应子节点，
        // 则指向 fail 节点的对应子节点
        node.children[ch] = node.fail
          ? (node.fail.children[ch] || this.root)
          : this.root;
      }
    }
  }
}
  ```

  ## 交互演示：多模式匹配

  AC 自动机可以同时匹配多个模式串：


    he
she
his
hers
    ushersheishis
