# 13 - AC 自动机基础 (Aho-Corasick)

  ## 什么是 AC 自动机？

  Aho-Corasick 自动机（AC 自动机）是 Alfred V. Aho 和 Margaret J. Corasick 于 1975 年提出的**多模式字符串匹配算法**。它结合了 Trie 树和 KMP 的思想，能在 O(n) 时间内同时查找多个模式串在文本中的出现位置。


>
    **核心思想：**

      - 将所有模式串构建成一棵 **Trie 树**

      - 在 Trie 树上添加**失败指针**（fail pointer），类似于 KMP 的 π 函数

      - 在文本串上匹配时，当前字符失配时沿着失败指针跳转，而**不需要回溯**文本指针




  ## AC 自动机的三个组成部分



    ****
    ********
    ****
  | 组成部分 | 描述 | 类比 |
| --- | --- | --- |
| Trie 树 | 存储所有模式串的字典树 | 模式串的集合表示 |
| 失败指针 | 当某个节点匹配失败时，跳转到另一个节点继续匹配。指向的是当前节点代表的字符串的 | 最长后缀 |
| 输出函数 | 当到达某个节点时，输出所有匹配到的模式串。通常包括该节点本身的匹配和沿着失败指针链路上的匹配 | 模式串匹配结果 |

  ## 失败指针（Fail Pointer）详解

  失败指针是 AC 自动机的精髓。假设当前节点对应字符串 S，则 fail 指针指向：**Trie 中存在的、S 的最长真后缀所对应的节点**。

  **构建过程（BFS）：**


    - 根节点的所有直接子节点的 fail 指针指向根节点

    - 使用 BFS 遍历 Trie 树：对于当前节点 u 及其子节点 v（对应字符 c）：

        - 如果 v 存在，则 `v.fail = u.fail.children[c]`（如果存在）否则继续沿着 fail 链查找，直到根节点

        - 如果 v 不存在，则将 `u.children[c]` 指向 `u.fail.children[c]`（转移表优化）







```javascript

// 构建失败指针的伪代码（BFS）
buildFailPointer():
  queue = new Queue()
  for each child of root:
    child.fail = root
    queue.push(child)

  while queue is not empty:
    u = queue.pop()
    for each (c, v) in u.children:
      // 计算 v 的失败指针
      f = u.fail
      while f != root and c not in f.children:
        f = f.fail
      if c in f.children:
        v.fail = f.children[c]
      else:
        v.fail = root
      // 合并输出（重要优化）
      v.output = v.output ∪ v.fail.output
      queue.push(v)
  ```

  ## 匹配过程

  AC 自动机的匹配过程与 KMP 高度相似，只是从单模式推广到了多模式：


```javascript

// AC 自动机匹配过程
match(text):
  node = root
  for i = 0 to text.length - 1:
    c = text[i]
    // 沿着失败指针查找可匹配的节点
    while node != root and c not in node.children:
      node = node.fail
    if c in node.children:
      node = node.children[c]
    // 检查当前节点是否有输出（匹配到的模式串）
    if node.output is not empty:
      for each pattern in node.output:
        print "位置", i - pattern.length + 1, ":", pattern
    // 注意：即使当前节点没有输出，也需要检查 node.output
    // 因为沿着 fail 链可能有其他匹配
  ```

  ## 复杂度分析







  | 指标 | 值 |
| --- | --- |
| 构建 Trie 树 | O(∑m) — 所有模式串的总长度 |
| 构建失败指针 | O(∑m) — BFS 遍历所有节点 |
| 匹配时间 | O(n + total_matches) — 线性于文本长度 + 匹配输出数 |
| 空间复杂度 | O(∑m · Σ) — 节点数 × 字符集大小（可用哈希表优化） |


>
    匹配的总时间复杂度为 O(n + m + z)，其中 z 是匹配输出的总次数。在最坏情况下（如模式串都是单个字符），z 可能达到 O(n·k)。


  ## 应用场景


    - **敏感词过滤** — 内容审核系统（如评论过滤、聊天监控）

    - **入侵检测** — Snort 等 IDS/IPS 系统中的特征匹配

    - **病毒扫描** — 在文件中查找病毒特征码

    - **生物信息学** — DNA 序列中查找多个基因片段

    - **自然语言处理** — 词典分词（最大匹配法）



  ## 图示：AC 自动机示例

  模式串集合: {"he", "she", "his", "hers"}


```

          root
         /    \
        h      s
       /      / \
      e      h   e
     / \    /    |
 (he)  r  (his) (she)
      /    ↑
     s    fail指针
    /   (虚线箭头)
 (hers)

 失败指针示例:
   "she" → "he" (最长公共后缀)
   "his" → "is"不存在 → "s"不存在 → root
   "hers" → "ers"不存在 → "rs"不存在
           → "s"不存在 → root
  ```

  ## 交互演示：概念理解


>
    AC 自动机的核心优势在于：

      - 扫描一遍文本即可找出**所有**模式串的匹配

      - 文本指针**永不回退**

      - 匹配时间与模式串数量**无关**（仅与总长度有关）


    具体实现和演示请见下一节：**AC 自动机实现**



  ## AC 自动机匹配演示

  模式串: he, she, his, hers   文本: "ushers"
