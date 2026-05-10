# Elasticsearch实战


## Elasticsearch 实战


搜索引擎Elasticsearch全文检索


Elasticsearch 是基于 Lucene 的分布式搜索和分析引擎，广泛应用于全文检索、日志分析和实时监控场景。


## Elasticsearch 架构


ES 采用分布式架构，通过分片和副本实现水平扩展和高可用。


```
┌────────────────────────────────────────────────────────────┐
│                  Elasticsearch 集群架构                      │
│                                                            │
│  Cluster（集群）                                            │
│  └── Index（索引）                                          │
│      └── Shard（分片）                                      │
│          ├── Primary Shard（主分片）                         │
│          └── Replica Shard（副本分片）                       │
│                                                            │
│  节点角色：                                                  │
│  ├── Master Node：管理集群状态、索引创建/删除                 │
│  ├── Data Node：存储数据、执行 CRUD 和搜索                    │
│  ├── Coordinating Node：接收请求、路由、聚合结果              │
│  └── Ingest Node：数据预处理（Pipeline）                     │
│                                                            │
│  分片策略：                                                  │
│  ├── 主分片数在创建索引时指定，之后不可修改                    │
│  ├── 副本数可随时调整                                        │
│  ├── 分片大小建议 10GB-50GB                                  │
│  └── 分片数 = 数据总量 / 目标分片大小                         │
└────────────────────────────────────────────────────────────┘
```


> **Note:** ES 7.x 以后默认每个索引 1 个主分片，应根据数据量和查询负载合理规划分片数。


## Query DSL 查询语法


ES 使用基于 JSON 的 Query DSL 进行灵活的查询构建。


```
查询上下文 vs 过滤上下文：
- Query Context：计算相关性评分（_score）
- Filter Context：只判断是否匹配，不计算评分（可缓存）

全文检索查询：
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "智能手机" } }
      ],
      "filter": [
        { "range": { "price": { "gte": 1000, "lte": 5000 } } },
        { "term": { "status": "active" } }
      ],
      "should": [
        { "match": { "brand": "苹果" } }
      ],
      "must_not": [
        { "term": { "is_deleted": true } }
      ]
    }
  }
}

精确匹配：
{ "term": { "status": "active" } }        // 精确值匹配
{ "terms": { "category": ["A", "B"] } }   // 多值匹配

范围查询：
{ "range": { "price": { "gte": 100, "lte": 500, "boost": 2.0 } } }

短语匹配：
{ "match_phrase": { "content": "机器学习算法" } }
```


## 聚合（Aggregations）分析


ES 聚合提供强大的数据分析能力，支持多层次嵌套聚合。


```
聚合三大类：
┌──────────────────────────────────────────────────────┐
│ 桶聚合（Bucket Aggregations）                         │
│ 将文档分组到不同的桶中                                  │
│                                                      │
│ GET /sales/_search                                   │
│ {                                                    │
│   "size": 0,                                         │
│   "aggs": {                                          │
│     "by_category": {                                 │
│       "terms": { "field": "category" },             │
│       "aggs": {                                      │
│         "avg_price": {                               │
│           "avg": { "field": "price" }               │
│         }                                            │
│       }                                              │
│     },                                               │
│     "price_ranges": {                                │
│       "range": {                                     │
│         "field": "price",                            │
│         "ranges": [                                  │
│           { "to": 50 },                              │
│           { "from": 50, "to": 100 },                │
│           { "from": 100 }                            │
│         ]                                            │
│       }                                              │
│     }                                                │
│   }                                                  │
│ }                                                    │
├──────────────────────────────────────────────────────┤
│ 指标聚合（Metric Aggregations）                       │
│ 计算数值指标                                          │
│ - avg / sum / min / max / cardinality                │
│ - stats（一次性计算 count/min/max/avg/sum）           │
│ - percentiles / percentile_ranks                     │
├──────────────────────────────────────────────────────┤
│ 管道聚合（Pipeline Aggregations）                     │
│ 对其他聚合结果进行再计算                                │
│ - bucket_sort：排序桶结果                             │
│ - bucket_selector：筛选桶                             │
│ - derivative / cumulative_sum                        │
└──────────────────────────────────────────────────────┘
```


## 分析器（Analyzer）


分析器决定了文本如何被分词和处理，直接影响搜索质量。


```
分析器组成：
┌─────────────────────────────────────────────────┐
│  Analyzer = Character Filter + Tokenizer +      │
│             Token Filter(s)                     │
│                                                 │
│  Character Filters（字符过滤器）                  │
│  ├── HTML Strip：去除 HTML 标签                  │
│  ├── Mapping：字符替换                          │
│  └── Pattern Replace：正则替换                   │
│                                                 │
│  Tokenizer（分词器）                             │
│  ├── standard：按单词边界分词                    │
│  ├── keyword：不分词，整体作为一个 token          │
│  ├── whitespace：按空白字符分词                  │
│  ├── pattern：按正则表达式分词                   │
│  └── icu_tokenizer：支持 Unicode 的分词          │
│                                                 │
│  Token Filters（词元过滤器）                      │
│  ├── lowercase：转小写                          │
│  ├── stop：去除停用词                           │
│  ├── synonym：同义词扩展                        │
│  ├── stemmer：词干提取                          │
│  └── edge_ngram：边缘 N-gram                    │
└─────────────────────────────────────────────────┘

中文分词：
- IK Analyzer：ik_smart（粗粒度）/ ik_max_word（细粒度）
- jieba：基于结巴分词的 ES 插件
- HanLP：功能丰富的中文 NLP 插件
```


## Mapping（映射）


Mapping 定义索引的字段结构和属性，类似于关系数据库的 Schema。


```
核心字段类型：
┌────────────┬──────────────────────────────────────┐
│ text       │ 全文检索，会分词，不支持排序和聚合       │
│ keyword    │ 精确匹配，支持排序和聚合，不分词         │
│ integer/long/float/double │ 数值类型              │
│ date       │ 日期类型，支持多种格式                  │
│ boolean    │ 布尔类型                              │
│ object     │ JSON 对象，内部扁平存储                │
│ nested     │ 对象数组，保持内部对象独立性            │
│ geo_point  │ 地理坐标                              │
└────────────┴──────────────────────────────────────┘

Mapping 示例：
PUT /products
{
  "mappings": {
    "properties": {
      "name": { "type": "text", "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart" },
      "category": { "type": "keyword" },
      "price": { "type": "float" },
      "tags": { "type": "keyword" },
      "created_at": { "type": "date", "format": "yyyy-MM-dd HH:mm:ss" },
      "comments": {
        "type": "nested",
        "properties": {
          "user": { "type": "keyword" },
          "content": { "type": "text" },
          "rating": { "type": "integer" }
        }
      }
    }
  }
}

Dynamic Mapping 策略：
- true：自动推断字段类型并添加（默认）
- false：忽略未知字段
- strict：遇到未知字段直接报错
```


> **Note:** 生产环境建议使用 strict 模式，避免动态映射导致字段类型不一致的问题。text + keyword 的 multi-field 是常用模式。


<!-- Converted from: 02_Elasticsearch实战.html -->
