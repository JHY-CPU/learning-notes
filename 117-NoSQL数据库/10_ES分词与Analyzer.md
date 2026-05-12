# Elasticsearch 分词与分析器

## 一、分析器内部原理

### 1.1 三阶段流水线

ES 中所有文本分析都遵循统一的流水线：

```
原文本 → [Character Filter] → [Tokenizer] → [Token Filter] → 词元流
```

每个阶段的职责：

| 阶段 | 职责 | 举例 |
|------|------|------|
| **Character Filter** | 在分词前对原始字符串做字符级变换（0个或多个） | `html_strip` 去除 HTML 标签；`mapping` 做字符映射（如把 `&` 映射为 `and`）；`pattern_replace` 正则替换 |
| **Tokenizer** | 将字符流切分为 Token 流（恰好一个） | `standard` 按空格+标点；`letter` 按非字母切分；`ngram`/`edge_ngram` 生成 N-gram |
| **Token Filter** | 对 Token 流做后处理（0个或多个） | `lowercase` 小写化；`stop` 移除停用词；`stemmer` 词干提取；`synonym` 同义词扩展 |

**底层机制**：Character Filter 在内存中构建一个 `CharFilter` 链表，每个 filter 维护一个 `Reader` → `Reader` 的映射。Tokenizer 使用 `java.io.Reader` 接口消费字符，通过状态机（如 `StandardTokenizer` 基于 Unicode 文本分割算法 UAX#29）产出 `AttributeSource`（包含 `CharTermAttribute`、`PositionIncrementAttribute` 等属性）。Token Filter 链式包装 `TokenStream`，每个 filter 重写 `incrementToken()` 方法。

### 1.2 Lucene 倒排索引中的位置信息

分词结果最终写入 Lucene 倒排索引，每个词项（Term）对应的 Postings 列表包含：

```
Term → [docID, termFreq, positions[], offsets[], payloads]
```

- **positions**：词在文档中的位置序号（用于短语查询 `match_phrase`）
- **offsets**：词在原文中的字符起止偏移（用于高亮）
- **payloads**：自定义载荷（极少使用）

**关键影响**：如果停用词过滤器移除了停用词，positions 数组会出现跳跃。这会导致 `match_phrase` 查询 "to be or not to be" 在移除停用词后可能匹配错误。解决办法：使用 `position_increment_gap` 或保留停用词位置（`enable_position_increments: true`）。

### 1.3 内置分析器详细对比

| 分析器 | Character Filter | Tokenizer | Token Filter | 典型场景 |
|--------|------------------|-----------|--------------|---------|
| **Standard** | 无 | `standard` | `standard`（含小写化）+ `stop`（可选） | 英文/通用 |
| **Simple** | 无 | `lowercase` | 无 | 不关心标点 |
| **Whitespace** | 无 | `whitespace` | 无 | 日志/命令行文本 |
| **Stop** | 无 | `standard` | `lowercase` + `stop` | 需要去停用词 |
| **Keyword** | 无 | `keyword` | `lowercase`（可选） | 精确匹配/不分词字段 |
| **Pattern** | 无 | `pattern`（默认 `\W+`） | `lowercase` + `stop`（可选） | 自定义分隔符 |
| **Language** | 无 | `standard` | 特定语言 stemmer + stop | 特定语言优化 |
| **Fingerprint** | 无 | `standard` | `lowercase` + `stop`（可选）+ `fingerprint`（排序去重） | 重复检测/指纹 |

```json
// 验证 Standard Analyzer 的分词效果
POST /_analyze
{
  "analyzer": "standard",
  "text": "The 2 QUICK Brown-Foxes jumped over the lazy dog's bone."
}
// 结果: ["the","2","quick","brown","foxes","jumped","over","the","lazy","dog's","bone"]
// 注意: 数字保留, 连字符作为分隔, 撇号不作为分隔
```

---

## 二、中文分词深度解析

### 2.1 中文分词的困难

中文没有天然的词边界（空格），分词本质上是一个**序列标注问题**（标注 B/M/E/S - 词首/词中/词尾/单字词）。主流方法：

1. **基于词典**：正向/逆向最大匹配（MMSeg），优点是速度快，缺点是未登录词识别差
2. **基于统计**：HMM/CRF，利用上下文概率解决歧义切分
3. **基于深度学习**：BiLSTM-CRF、BERT-based，精度最高但推理慢

### 2.2 IK 分析器详解

IK 是基于 Java 实现的中文分词器，内核采用 **词典 + 正向迭代最细粒度切分算法**：

```
输入: "中华人民共和国国歌"
ik_max_word: ["中华","中华人民","中华人民共和国","人民","人民共和国","共和国","国歌"]
ik_smart:    ["中华人民共和国","国歌"]
```

**ik_max_word**（索引时用）：穷尽所有可能的词组合，保证召回率。
**ik_smart**（搜索时用）：使用最粗粒度切分，保证精度。

**最佳实践**：索引用 `ik_max_word`，搜索用 `ik_smart`。这样索引覆盖面广，搜索时不会因为分词过细导致误匹配。

```json
// 安装 IK
// ./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.12.0/elasticsearch-analysis-ik-8.12.0.zip

// 创建索引并配置 IK
PUT /ecommerce
{
  "settings": {
    "analysis": {
      "analyzer": {
        "ik_index_analyzer": {
          "type": "custom",
          "tokenizer": "ik_max_word",
          "filter": ["lowercase"]
        },
        "ik_search_analyzer": {
          "type": "custom",
          "tokenizer": "ik_smart",
          "filter": ["lowercase"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "product_name": {
        "type": "text",
        "analyzer": "ik_index_analyzer",
        "search_analyzer": "ik_search_analyzer",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      }
    }
  }
}

// 测试分词
POST /ecommerce/_analyze
{
  "analyzer": "ik_index_analyzer",
  "text": "iPhone手机保护壳"
}
// 结果: ["iphone","手机","保护","保护壳","壳"]
```

**常见坑点**：
- IK 词典更新后需要重启 ES 或调用 `_analyze` 的 `reload` 接口，热更新仅支持远程词典（`remote_ext_dict`）
- 不要在同一个字段上混用 `ik_max_word` 和 `ik_smart`，会导致分词不一致
- IK 对英文数字的处理不如 `standard`，英文和数字会被直接作为整词保留

### 2.3 pinyin 分词器

用于支持「输入拼音也能搜到中文结果」的场景。

```json
PUT /pinyin_demo
{
  "settings": {
    "analysis": {
      "analyzer": {
        "pinyin_analyzer": {
          "tokenizer": "my_pinyin",
          "filter": ["lowercase"]
        }
      },
      "tokenizer": {
        "my_pinyin": {
          "type": "pinyin",
          "keep_first_letter": true,       // 保留首字母: "张三" -> "zs"
          "keep_separate_first_letter": false,
          "keep_full_pinyin": true,         // 保留全拼: "张三" -> "zhang,san"
          "keep_original": true,            // 保留原文
          "limit_first_letter_length": 16,
          "keep_spaces": true,
          "remove_duplicated_term": true,   // 去重
          "none_chinese_pinyin_tokenize": true
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "analyzer": "ik_max_word",
        "fields": {
          "pinyin": {
            "type": "text",
            "analyzer": "pinyin_analyzer",
            "search_analyzer": "pinyin_analyzer"
          }
        }
      }
    }
  }
}
```

**真实案例**：电商搜索中，用户输入 `zhangsan` 或 `zs` 也能匹配到「张三」。常见做法是将 `name` 字段同时建 IK 索引和 pinyin 索引，搜索时使用 `multi_match` 或 `bool` 查询组合两个字段的得分。

```python
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

# 搜索 "zhangsan" -> 同时搜 name 和 name.pinyin
result = es.search(index="pinyin_demo", body={
    "query": {
        "multi_match": {
            "query": "zhangsan",
            "fields": ["name^3", "name.pinyin"]
        }
    }
})
```

### 2.4 其他中文分词方案对比

| 分析器 | 算法 | 精度 | 速度 | 未登录词 | 维护状态 |
|--------|------|------|------|---------|---------|
| **IK** | 词典+MMSeg | 中 | 快 | 弱 | 社区活跃 |
| **jieba** (结巴) | HMM+词典 | 中 | 快 | 中 | Python 生态 |
| **HanLP** | 深度学习+规则 | 高 | 中 | 强 | 商业版活跃 |
| **THULAC** | CRF | 高 | 慢 | 强 | 学术维护 |
| **SmartChinese** | HMM | 中 | 快 | 弱 | ES 内置 |

---

## 三、自定义分析器进阶

### 3.1 完整构建示例

```json
PUT /advanced_custom
{
  "settings": {
    "analysis": {
      "char_filter": {
        "my_mapping": {
          "type": "mapping",
          "mappings": [
            "٠=>0", "١=>1", "٢=>2", "٣=>3", "٤=>4",
            "٥=>5", "٦=>6", "٧=>7", "٨=>8", "٩=>9"
          ]
        },
        "my_pattern": {
          "type": "pattern_replace",
          "pattern": "(\\d{3})\\d{4}(\\d{4})",
          "replacement": "$1****$2"
        }
      },
      "filter": {
        "my_synonym": {
          "type": "synonym",
          "synonyms_path": "analysis/synonyms.txt",
          "updateable": true
        },
        "my_edge_ngram": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 10
        },
        "my_shingle": {
          "type": "shingle",
          "min_shingle_size": 2,
          "max_shingle_size": 3,
          "output_unigrams": true
        },
        "my_stemmer_override": {
          "type": "stemmer_override",
          "rules": [
            "ran => run",
            "mice => mouse"
          ]
        }
      },
      "analyzer": {
        "autocomplete_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "my_edge_ngram"]
        },
        "shingle_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "my_shingle"]
        },
        "pii_analyzer": {
          "type": "custom",
          "char_filter": ["my_pattern"],
          "tokenizer": "keyword",
          "filter": ["lowercase"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "suggestion": {
        "type": "text",
        "analyzer": "autocomplete_analyzer",
        "search_analyzer": "standard"
      },
      "phrase": {
        "type": "text",
        "analyzer": "shingle_analyzer"
      },
      "phone": {
        "type": "text",
        "analyzer": "pii_analyzer"
      }
    }
  }
}
```

### 3.2 同义词方案深度对比

| 方案 | 加载时机 | 热更新 | 性能影响 | 适用场景 |
|------|---------|--------|---------|---------|
| **内联 synonyms** | 索引创建时 | 需要 reindex | 无额外 IO | 词汇表小且固定 |
| **synonyms_path 文件** | 索引创建时 | 需要 reindex | 一次读取 | 中等词汇表 |
| **updateable synonyms** | 运行时 | 支持热更新 | 查询时读取缓存 | 词汇表经常变化 |
| **index alias + 应用层扩展** | 查询时 | 即时生效 | 需要多次查询 | 同义词关系复杂 |

**常见坑点**：
- 同义词扩展只在**索引时**或**搜索时**生效。如果同义词在索引时扩展，新添加的同义词不会匹配已索引的文档（需要 reindex）。如果在搜索时扩展，不影响索引但查询变慢。
- 推荐：**索引时不做同义词扩展，搜索时通过 `search_analyzer` 加载同义词**，这样更新同义词文件后只需 close/open 索引即可生效。

```json
// 搜索时同义词方案
PUT /synonym_search
{
  "settings": {
    "analysis": {
      "filter": {
        "search_synonyms": {
          "type": "synonym",
          "synonyms": [
            "计算机,电脑,微机",
            "手机,移动电话,smartphone"
          ]
        }
      },
      "analyzer": {
        "index_analyzer": {
          "type": "custom",
          "tokenizer": "ik_max_word",
          "filter": ["lowercase"]
        },
        "search_analyzer": {
          "type": "custom",
          "tokenizer": "ik_smart",
          "filter": ["lowercase", "search_synonyms"]
        }
      }
    }
  }
}
```

### 3.3 Edge N-gram 与 Completion Suggester 对比

| 方案 | 原理 | 延迟 | 内存占用 | 场景 |
|------|------|------|---------|------|
| **Edge N-gram 索引** | 将每个词拆成前缀子串存入倒排索引 | 低（直接搜倒排） | 高（索引膨胀 3-5x） | 中等数据量前缀补全 |
| **Completion Suggester** | 基于 FST（Finite State Transducer） | 极低（O(len(prefix))） | 中（FST 压缩） | 大量补全建议 |
| **Search-as-you-type** | ES 7.x 新类型，内置 `edge_ngram` + `shingle` | 低 | 高 | 前缀搜索 |

```json
// Completion Suggester 示例
PUT /products
{
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "ik_max_word" },
      "suggest": {
        "type": "completion",
        "analyzer": "simple",
        "search_analyzer": "simple",
        "contexts": [
          { "name": "category", "type": "category" }
        ]
      }
    }
  }
}

POST /products/_doc
{
  "name": "iPhone 15 Pro Max",
  "suggest": {
    "input": ["iPhone", "iPhone 15", "iPhone 15 Pro Max"],
    "weight": 10,
    "contexts": {
      "category": ["手机", "苹果"]
    }
  }
}

// 查询补全
POST /products/_search
{
  "suggest": {
    "product_suggest": {
      "prefix": "iph",
      "completion": {
        "field": "suggest",
        "size": 5,
        "contexts": {
          "category": ["手机"]
        }
      }
    }
  }
}
```

**常见坑点**：
- 不要用 `wildcard` 做前缀查询（如 `*keyword`），性能极差（需要遍历所有 term）。前缀查询用 `prefix` 或 `completion suggester`
- `wildcard` 以通配符开头时（`*xxx`），无法利用倒排索引的跳表优化，退化为全量扫描
- Completion Suggester 不支持 fuzzy 模糊匹配，需要 fuzzy 的场景使用 `phrase suggester`

---

## 四、分析器性能分析

### 4.1 分词性能基准

在 100 万文档（平均 500 字/文档）上的分词耗时对比：

| 分析器 | 索引耗时 | 索引大小 | 查询延迟（match） |
|--------|---------|---------|------------------|
| **standard** | 基准 1x | 基准 1x | ~5ms |
| **ik_max_word** | 1.3x | 1.8x | ~8ms |
| **ik_smart** | 1.1x | 1.2x | ~6ms |
| **pinyin** | 1.5x | 2.5x | ~10ms |
| **edge_ngram(2-10)** | 2.0x | 4.0x | ~3ms（前缀查询） |

**结论**：
- `ik_max_word` 索引膨胀严重，但搜索时使用 `ik_smart` 可以补偿
- pinyin 索引膨胀约 2.5 倍，是中文搜索「拼音+中文」方案的常见开销
- edge_ngram 索引膨胀最大（4x），适合数据量不大但需要前缀补全的场景

### 4.2 调试分析器的技巧

```json
// 1. 对比不同 analyzer 的分词结果
POST /_analyze
{
  "text": "中华人民共和国国歌",
  "explain": true,
  "analyzer": "ik_max_word"
}

// 2. 使用自定义 tokenizer + filter 逐步调试
POST /_analyze
{
  "text": "The QUICK Brown Foxes",
  "tokenizer": "standard",
  "char_filter": ["html_strip"],
  "filter": ["lowercase", "english_stemmer"]
}

// 3. 检查字段实际使用的分析器
GET /my_index/_mapping/field/content

// 4. 查看 term vectors（显示每个词的 TF、positions、offsets）
GET /my_index/_termvectors/1
{
  "fields": ["content"],
  "positions": true,
  "offsets": true
}
```

### 4.3 实际案例：电商搜索分析器设计

```
用户搜索: "iPhone手机"
期望匹配: iPhone手机保护壳、苹果手机、apple手机壳
```

```json
PUT /ecommerce_v2
{
  "settings": {
    "analysis": {
      "char_filter": {
        "emoticons": {
          "type": "mapping",
          "mappings": [":) => _happy_", ":(" => _sad_"]
        }
      },
      "filter": {
        "cn_synonyms": {
          "type": "synonym",
          "synonyms": [
            "电脑,计算机,PC",
            "手机,移动电话,smartphone",
            "苹果,apple,apple inc"
          ]
        },
        "cn_stop": {
          "type": "stop",
          "stopwords": "_chinese_"
        },
        "pinyin_filter": {
          "type": "pinyin",
          "keep_first_letter": true,
          "keep_full_pinyin": true,
          "keep_original": true
        }
      },
      "analyzer": {
        "ik_synonym": {
          "type": "custom",
          "tokenizer": "ik_max_word",
          "filter": ["lowercase", "cn_synonyms"]
        },
        "ik_search": {
          "type": "custom",
          "tokenizer": "ik_smart",
          "filter": ["lowercase", "cn_synonyms"]
        },
        "pinyin_index": {
          "type": "custom",
          "tokenizer": "ik_max_word",
          "filter": ["lowercase", "pinyin_filter"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "analyzer": "ik_synonym",
        "search_analyzer": "ik_search",
        "fields": {
          "keyword": { "type": "keyword" },
          "pinyin": {
            "type": "text",
            "analyzer": "pinyin_index"
          }
        }
      },
      "suggest": {
        "type": "completion",
        "analyzer": "simple"
      }
    }
  }
}
```

搜索时组合查询：

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def search_products(keyword, category=None):
    """电商搜索：组合 IK + pinyin + suggester"""
    body = {
        "query": {
            "bool": {
                "should": [
                    # IK 中文分词（权重最高）
                    {"match": {"name": {"query": keyword, "boost": 3}}},
                    # 拼音匹配
                    {"match": {"name.pinyin": {"query": keyword, "boost": 1}}},
                ],
                "minimum_should_match": 1
            }
        },
        "suggest": {
            "product_suggest": {
                "prefix": keyword,
                "completion": {
                    "field": "suggest",
                    "size": 5,
                    "fuzzy": {"fuzziness": "AUTO"}
                }
            }
        },
        "highlight": {
            "fields": {"name": {}}
        }
    }
    if category:
        body["query"]["bool"]["filter"] = [
            {"term": {"category.keyword": category}}
        ]
    return es.search(index="ecommerce_v2", body=body)
```

---

## 五、常见问题与排错

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 搜索「电脑」匹配不到「计算机」 | 未配置同义词 | 在 search_analyzer 中加入 synonym filter |
| 中文搜索「中华人民」只匹配到一个文档 | 索引时分词过粗 | 索引用 `ik_max_word`，搜索用 `ik_smart` |
| pinyin 搜索「zhangsan」无结果 | 首字母模式未开启 | `keep_first_letter: true, keep_full_pinyin: true` |
| match_phrase 查询结果为空 | 停用词移除导致 position 跳跃 | 避免在 phrase 查询字段上使用 stop filter |
| 索引后搜不到新词 | 同义词只在 search_analyzer 配置 | 用 search_analyzer 方案，close/open 索引刷新 |
| wildcard 查询 `*xxx` 超时 | 无法利用倒排索引 | 改用 completion suggester 或 ngram |
