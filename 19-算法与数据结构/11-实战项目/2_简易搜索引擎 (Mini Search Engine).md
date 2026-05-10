# 简易搜索引擎 (Mini Search Engine)

## 项目需求与功能分析

搜索引擎是信息检索的核心应用。本项目从零构建一个简易搜索引擎，实现文档索引、查询检索、结果排序等核心功能，深入理解倒排索引、TF-IDF 和 BM25 等经典算法。

### 核心功能

- 文档预处理（分词、去停用词）
- 倒排索引构建与维护
- TF-IDF 相似度计算
- BM25 排序算法
- 布尔查询支持（AND / OR / NOT）
- 查询结果高亮显示

### 系统架构

```
文档集合 -> 分词器 -> 倒排索引 -> 检索引擎 -> 排序算法 -> 结果展示
```

## 核心算法原理

### 倒排索引 (Inverted Index)

将文档从 `文档 -> 词` 反转为 `词 -> 文档列表`：

```python
{
  "算法": [(doc1, [3, 15]), (doc2, [7])],
  "结构": [(doc1, [5]), (doc3, [2, 9])]
}
```

### TF-IDF

- **TF (词频)**: 词在文档中出现的频率
- **IDF (逆文档频率)**: log(N / df)，衡量词的区分度
- **TF-IDF** = TF x IDF

### BM25

BM25 引入文档长度归一化和饱和函数，是 TF-IDF 的改进版本：

```
score(D,Q) = sum(IDF(qi) * f(qi,D)*(k1+1) / (f(qi,D) + k1*(1-b+b*|D|/avgdl)))
```

常用参数: k1=1.2, b=0.75

## 完整代码实现

```python
import math, re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional


@dataclass
class Document:
    doc_id: int; title: str; content: str
    tokens: List[str] = field(default_factory=list)


class Tokenizer:
    STOPWORDS = set("的 了 在 是 我 有 和 就 不 人 都 一 个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这".split())
    EN_STOPWORDS = set("the a an is are was were be been have has had do does did will would shall should may might can could of in to for on with at by from as and but or nor not no so if then than too very just about up out".split())

    def tokenize(self, text):
        text = text.lower()
        chinese = re.findall(r'[一-鿿]+', text)
        english = re.findall(r'[a-z]+', text)
        tokens = []
        for seg in chinese:
            chars = list(seg); tokens.extend(chars)
            for i in range(len(chars)-1): tokens.append(chars[i]+chars[i+1])
        tokens.extend(english)
        return [t for t in tokens if t not in self.STOPWORDS and t not in self.EN_STOPWORDS and len(t)>0]


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[Tuple[int,List[int]]]] = defaultdict(list)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.total_docs = 0
        self.doc_lengths: Dict[int, int] = {}

    def add_document(self, doc):
        self.total_docs += 1
        self.doc_lengths[doc.doc_id] = len(doc.tokens)
        tp = defaultdict(list)
        for pos, tok in enumerate(doc.tokens): tp[tok].append(pos)
        for term, positions in tp.items():
            self.index[term].append((doc.doc_id, positions))
            self.doc_freq[term] += 1


class SearchEngine:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.idx = InvertedIndex()
        self.docs: Dict[int, Document] = {}
        self.next_id = 0

    def add_document(self, title, content):
        doc = Document(self.next_id, title, content)
        doc.tokens = self.tokenizer.tokenize(title + " " + content)
        self.docs[doc.doc_id] = doc; self.idx.add_document(doc)
        self.next_id += 1; return doc.doc_id

    def add_documents(self, pairs):
        return [self.add_document(t, c) for t, c in pairs]

    def _bm25_score(self, query_terms, doc_id, k1=1.2, b=0.75):
        doc = self.docs[doc_id]; dl = len(doc.tokens)
        avgdl = sum(self.idx.doc_lengths[d] for d in self.docs) / len(self.docs)
        score = 0.0
        for term in query_terms:
            tf = doc.tokens.count(term)
            df = self.idx.doc_freq.get(term, 0)
            n = self.idx.total_docs
            idf = math.log((n - df + 0.5)/(df + 0.5) + 1)
            score += idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        return score

    def search_bm25(self, query, top_k=10):
        terms = self.tokenizer.tokenize(query)
        if not terms: return []
        candidates = set()
        for t in terms:
            for did, _ in self.idx.index.get(t, []): candidates.add(did)
        results = [(self._bm25_score(terms, did), self.docs[did]) for did in candidates]
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def search_boolean(self, query):
        tokens = query.split()
        must, must_not, should = set(), set(), set()
        for t in tokens:
            if t.startswith('+'): must.add(t[1:].lower())
            elif t.startswith('-'): must_not.add(t[1:].lower())
            else: should.add(t.lower())
        result = None
        for term in must:
            ts = self.tokenizer.tokenize(term)
            docs = set(did for t in ts for did,_ in self.idx.index.get(t,[]))
            result = docs if result is None else result & docs
        if should:
            or_docs = set(did for t in should for tt in self.tokenizer.tokenize(t) for did,_ in self.idx.index.get(tt,[]))
            result = or_docs if result is None else result | or_docs
        if result is None: result = set()
        for term in must_not:
            for tt in self.tokenizer.tokenize(term):
                for did,_ in self.idx.index.get(tt,[]): result.discard(did)
        return result

    def highlight(self, text, query):
        result = text
        for term in self.tokenizer.tokenize(query):
            result = re.sub(f'({re.escape(term)})', r'\033[91m\1\033[0m', result, flags=re.IGNORECASE)
        return result
```

## 测试用例

```python
import unittest

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        self.engine.add_documents([
            ("Python 教程", "Python 是一种编程语言，简单易学"),
            ("Java 教程", "Java 是一种面向对象的编程语言"),
            ("算法入门", "学习排序算法和搜索算法"),
            ("数据结构", "数组 链表 栈 队列 树 图"),
        ])

    def test_search_returns_results(self):
        results = self.engine.search_bm25("Python")
        self.assertGreater(len(results), 0)

    def test_boolean_and(self):
        ids = self.engine.search_boolean("+Python +编程")
        self.assertIn(0, ids)

    def test_boolean_not(self):
        ids = self.engine.search_boolean("编程 -Python")
        self.assertNotIn(0, ids); self.assertIn(1, ids)

    def test_empty_query(self):
        self.assertEqual(len(self.engine.search_bm25("")), 0)

    def test_not_in_index(self):
        self.assertEqual(len(self.engine.search_bm25("量子计算")), 0)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **中文分词**：集成 jieba 等专业中文分词库
2. **查询纠错**：基于编辑距离的拼写纠错
3. **查询建议**：基于搜索日志的查询补全
4. **PageRank**：加入网页权威度排序因子
5. **分布式索引**：将索引分片存储在多个节点
6. **近实时索引**：支持动态添加和删除文档
7. **语义搜索**：使用词向量实现语义级别检索
