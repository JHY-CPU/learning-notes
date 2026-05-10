# 拼写检查器 (Spell Checker)

## 项目需求与功能分析

拼写检查是文字处理软件的基础功能。本项目实现基于编辑距离和 BK-Tree 的拼写检查器，支持高效的拼写纠错和候选词推荐。

### 核心功能

- 词典管理（添加 / 删除单词）
- 拼写检查（判断单词是否正确）
- 纠错建议（推荐最可能的正确拼写）
- 编辑距离计算
- BK-Tree 高效索引

### 应用场景

- 文本编辑器拼写检查
- 搜索引擎查询纠错
- 输入法联想
- 自然语言处理预处理

## 核心算法原理

### 编辑距离 (Levenshtein Distance)

将字符串 A 转换为字符串 B 所需的最少单字符操作次数（插入、删除、替换）。

动态规划递推公式：
```
dp[i][j] = dp[i-1][j-1]                    if A[i]==B[j]
dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  otherwise
```

### BK-Tree

利用编辑距离的度量性质构建的树结构：
- 每个节点存储一个单词
- 子节点按编辑距离分组
- 查找时利用三角不等式剪枝：若 d(query, node) = k，则只搜索 d 在 [t-k, t+k] 范围内的子节点

### 概率纠错

根据编辑距离和词频计算候选词概率：
```
P(correction | word) ∝ P(word | correction) * P(correction)
```

## 完整代码实现

```python
from collections import Counter
from typing import List, Tuple, Dict, Optional
import re


def edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


class BKTreeNode:
    """BK-Tree 节点"""
    def __init__(self, word: str):
        self.word = word
        self.children: Dict[int, 'BKTreeNode'] = {}


class BKTree:
    """BK-Tree - 基于编辑距离的索引树"""

    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0

    def insert(self, word: str):
        if self.root is None:
            self.root = BKTreeNode(word)
            self.size = 1
            return

        node = self.root
        d = edit_distance(word, node.word)
        while d in node.children:
            node = node.children[d]
            d = edit_distance(word, node.word)
        node.children[d] = BKTreeNode(word)
        self.size += 1

    def search(self, word: str, max_dist: int = 2) -> List[Tuple[int, str]]:
        """查找编辑距离 <= max_dist 的所有词"""
        if self.root is None:
            return []

        results = []
        self._search(self.root, word, max_dist, results)
        results.sort()
        return results

    def _search(self, node: BKTreeNode, word: str, max_dist: int, results: list):
        d = edit_distance(word, node.word)
        if d <= max_dist:
            results.append((d, node.word))

        # 利用三角不等式剪枝
        for dist in range(max(d - max_dist, 0), d + max_dist + 1):
            if dist in node.children:
                self._search(node.children[dist], word, max_dist, results)


class SpellChecker:
    """拼写检查器"""

    def __init__(self):
        self.word_freq: Counter = Counter()
        self.total_words = 0
        self.bk_tree = BKTree()

    def load_dictionary(self, words: List[str]):
        """加载词典"""
        for word in words:
            w = word.lower().strip()
            if w.isalpha():
                self.word_freq[w] += 1
                self.total_words += 1
        # 构建 BK-Tree
        for word in self.word_freq:
            self.bk_tree.insert(word)

    def load_text(self, text: str):
        """从文本中学习词典"""
        words = re.findall(r'[a-zA-Z]+', text.lower())
        self.load_dictionary(words)

    def is_correct(self, word: str) -> bool:
        """检查单词是否正确"""
        return word.lower() in self.word_freq

    def suggest(self, word: str, max_dist: int = 2, top_k: int = 5) -> List[Tuple[str, float]]:
        """推荐纠正候选词"""
        word = word.lower()
        candidates = self.bk_tree.search(word, max_dist)

        if not candidates:
            return []

        # 按概率排序: P(correction|word) ∝ P(word|correction) * P(correction)
        scored = []
        for dist, candidate in candidates:
            if dist == 0:
                continue  # 跳过完全匹配
            # 编辑距离越小越可能 (P(word|correction))
            edit_prob = 1.0 / (dist + 1)
            # 词频越高越可能 (P(correction))
            freq_prob = self.word_freq[candidate] / self.total_words
            score = edit_prob * freq_prob
            scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def correct_text(self, text: str) -> str:
        """纠正整段文本"""
        words = re.findall(r'[a-zA-Z]+|[^a-zA-Z]+', text)
        result = []
        for part in words:
            if part.isalpha() and not self.is_correct(part):
                suggestions = self.suggest(part, top_k=1)
                if suggestions:
                    result.append(suggestions[0][0])
                else:
                    result.append(part)
            else:
                result.append(part)
        return ''.join(result)

    def check_document(self, text: str) -> List[Tuple[str, int, List[Tuple[str, float]]]]:
        """检查文档，返回所有疑似错误"""
        errors = []
        for match in re.finditer(r'[a-zA-Z]+', text):
            word = match.group()
            if not self.is_correct(word):
                suggestions = self.suggest(word)
                errors.append((word, match.start(), suggestions))
        return errors
```

## 测试用例

```python
import unittest

class TestSpellChecker(unittest.TestCase):
    def setUp(self):
        self.sc = SpellChecker()
        self.sc.load_dictionary([
            'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog',
            'apple', 'application', 'apply', 'happy', 'hello', 'world',
            'python', 'programming', 'algorithm', 'data', 'structure',
        ])

    def test_correct_word(self):
        self.assertTrue(self.sc.is_correct('hello'))
        self.assertTrue(self.sc.is_correct('python'))
        self.assertFalse(self.sc.is_correct('helo'))

    def test_edit_distance(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('', 'abc'), 3)
        self.assertEqual(edit_distance('abc', 'abc'), 0)

    def test_suggest(self):
        suggestions = self.sc.suggest('helo', top_k=3)
        words = [w for w, _ in suggestions]
        self.assertIn('hello', words)

    def test_correct_text(self):
        corrected = self.sc.correct_text('helo wrold')
        self.assertIn('hello', corrected)
        self.assertIn('world', corrected)

    def test_bk_tree(self):
        tree = BKTree()
        for w in ['cat', 'car', 'card', 'cart', 'dog']:
            tree.insert(w)
        results = tree.search('car', max_dist=1)
        words = [w for _, w in results]
        self.assertIn('car', words)
        self.assertIn('card', words)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **拼写自动补全**：前缀匹配 + 词频排序
2. **语音纠错**：基于发音相似度的纠错（Soundex / Metaphone）
3. **上下文纠错**：利用 N-gram 语言模型判断词序是否合理
4. **多语言支持**：扩展到中文错别字检查
5. **学习用户词典**：记住用户常用的专业术语
6. **模糊搜索**：在搜索引擎中集成拼写纠错
7. **性能优化**：SymSpell 算法实现 O(1) 查找
