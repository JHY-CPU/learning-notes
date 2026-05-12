# ES 查询 DSL

## 1. Query DSL 概述

Elasticsearch 使用基于 JSON 的 Query DSL（Domain Specific Language）进行查询。查询分为两种上下文：

- **查询上下文（Query Context）**：计算文档与查询的相关性得分（_score）
- **过滤上下文（Filter Context）**：只判断是否匹配，不计算得分，结果可缓存

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "password"))

# 基本查询
result = es.search(
    index="products",
    body={"query": {"match": {"name": "手机"}}}
)
```

## 2. 全文查询

### match 查询

`match` 查询是最常用的全文查询，会对查询文本进行分词后搜索。

```python
# 基本 match 查询
result = es.search(
    index="articles",
    body={
        "query": {
            "match": {
                "content": "Elasticsearch 搜索引擎"
            }
        }
    }
)

# match 查询带参数
result = es.search(
    index="articles",
    body={
        "query": {
            "match": {
                "content": {
                    "query": "Elasticsearch 搜索引擎",
                    "operator": "and",               # 所有词都必须出现
                    "minimum_should_match": "75%"     # 至少 75% 的词匹配
                }
            }
        }
    }
)

# 提取结果
for hit in result["hits"]["hits"]:
    print(f"文档ID: {hit['_id']}, 得分: {hit['_score']}")
    print(f"内容: {hit['_source']['content'][:100]}")
```

### match_phrase 查询

`match_phrase` 要求所有词项按顺序出现，且位置相邻。

```python
# 短语匹配（允许一定的位置偏差）
result = es.search(
    index="articles",
    body={
        "query": {
            "match_phrase": {
                "content": {
                    "query": "分布式搜索引擎",
                    "slop": 2  # 允许词项之间有 2 个位置的偏差
                }
            }
        }
    }
)
```

### multi_match 查询

在多个字段上执行 match 查询。

```python
# 多字段查询，使用 ^ 指定权重
result = es.search(
    index="articles",
    body={
        "query": {
            "multi_match": {
                "query": "Elasticsearch",
                "fields": ["title^3", "content", "tags^2"],
                "type": "best_fields"
            }
        }
    }
)
```

type 选项说明：
- **best_fields**：使用得分最高的字段（默认）
- **most_fields**：合并所有字段的得分
- **cross_fields**：将所有字段视为一个大字段
- **phrase**：在每个字段上执行 match_phrase

### query_string 查询

支持复杂查询语法（AND/OR/NOT/括号）。

```python
result = es.search(
    index="articles",
    body={
        "query": {
            "query_string": {
                "query": "(Elasticsearch OR Solr) AND 分布式 NOT 单机",
                "fields": ["title", "content"]
            }
        }
    }
)
```

## 3. 精确值查询

### term 查询

`term` 查询用于精确匹配，不进行分词处理。

```python
# 精确匹配 keyword 字段
result = es.search(
    index="products",
    body={
        "query": {
            "term": {
                "category": "electronics"
            }
        }
    }
)

# 带权重的 term 查询
result = es.search(
    index="products",
    body={
        "query": {
            "term": {
                "status": {"value": "active", "boost": 2.0}
            }
        }
    }
)
```

### terms 查询

匹配多个值中的任意一个（类似 SQL 的 IN）。

```python
result = es.search(
    index="products",
    body={
        "query": {
            "terms": {
                "category": ["electronics", "books", "clothing"]
            }
        }
    }
)
```

### range 查询

范围查询，适用于数值、日期等类型。

```python
# 数值范围
result = es.search(
    index="products",
    body={
        "query": {
            "range": {
                "price": {"gte": 100, "lte": 500}
            }
        }
    }
)

# 日期范围
result = es.search(
    index="orders",
    body={
        "query": {
            "range": {
                "created_at": {
                    "gte": "2024-01-01",
                    "lte": "2024-12-31",
                    "format": "yyyy-MM-dd"
                }
            }
        }
    }
)

# 相对日期范围
result = es.search(
    index="orders",
    body={
        "query": {
            "range": {
                "created_at": {
                    "gte": "now-30d/d",   # 30 天前
                    "lte": "now/d"         # 今天
                }
            }
        }
    }
)
```

### exists 和 ids 查询

```python
# exists: 字段是否存在
result = es.search(
    index="users",
    body={"query": {"exists": {"field": "email"}}}
)

# ids: 按文档 ID 查询
result = es.search(
    index="products",
    body={"query": {"ids": {"values": ["product_1", "product_2"]}}}
)
```

## 4. 组合查询（Bool 查询）

Bool 查询是最强大的组合查询，包含四种子句：

```python
result = es.search(
    index="products",
    body={
        "query": {
            "bool": {
                "must": [              # 必须匹配（影响得分）
                    {"match": {"name": "手机"}}
                ],
                "filter": [            # 必须匹配（不影响得分，可缓存）
                    {"term": {"category": "electronics"}},
                    {"range": {"price": {"gte": 1000, "lte": 5000}}}
                ],
                "should": [            # 应该匹配（加分项）
                    {"term": {"brand": "apple"}},
                    {"term": {"brand": "samsung"}}
                ],
                "must_not": [          # 必须不匹配
                    {"term": {"status": "out_of_stock"}}
                ]
            }
        }
    }
)
```

### filter vs must 的区别

```python
# filter（推荐用于精确条件）：不计算得分，结果可缓存，性能更好
result = es.search(
    index="products",
    body={
        "query": {
            "bool": {
                "filter": [
                    {"term": {"status": "active"}},
                    {"range": {"price": {"lte": 100}}}
                ]
            }
        }
    }
)

# must（用于需要相关性排序的场景）：计算得分
result = es.search(
    index="products",
    body={
        "query": {
            "bool": {
                "must": [
                    {"match": {"name": "智能手机"}}
                ],
                "should": [
                    {"term": {"brand": "apple"}},
                    {"term": {"brand": "samsung"}}
                ],
                "minimum_should_match": 1  # should 子句至少匹配 1 个
            }
        }
    }
)
```

## 5. 排序与分页

```python
# 按字段排序
result = es.search(
    index="products",
    body={
        "query": {"match_all": {}},
        "sort": [
            {"price": {"order": "asc"}},
            {"created_at": {"order": "desc"}},
            "_score"
        ]
    }
)

# 按脚本排序
result = es.search(
    index="products",
    body={
        "query": {"match_all": {}},
        "sort": {
            "_script": {
                "type": "number",
                "script": {
                    "source": "doc['price'].value * params.rate",
                    "params": {"rate": 0.8}
                },
                "order": "asc"
            }
        }
    }
)

# 分页（from + size）
page = 2
size = 20
result = es.search(
    index="products",
    body={
        "from": (page - 1) * size,
        "size": size,
        "query": {"match_all": {}}
    }
)
```

## 6. 聚合查询

### 度量聚合

```python
result = es.search(
    index="products",
    body={
        "size": 0,  # 不返回文档，只返回聚合结果
        "aggs": {
            "price_stats": {
                "stats": {"field": "price"}  # 返回 count/min/max/avg/sum
            },
            "max_price": {
                "max": {"field": "price"}
            },
            "avg_rating": {
                "avg": {"field": "rating"}
            }
        }
    }
)
print(f"价格统计: {result['aggregations']['price_stats']}")
```

### 桶聚合

```python
# 按类别分组并计算子聚合
result = es.search(
    index="products",
    body={
        "size": 0,
        "aggs": {
            "by_category": {
                "terms": {
                    "field": "category",
                    "size": 10,            # 返回前 10 个桶
                    "min_doc_count": 5     # 最少 5 个文档
                },
                "aggs": {
                    "avg_price": {"avg": {"field": "price"}},
                    "max_price": {"max": {"field": "price"}}
                }
            }
        }
    }
)

for bucket in result["aggregations"]["by_category"]["buckets"]:
    print(f"类别: {bucket['key']}, 数量: {bucket['doc_count']}, "
          f"均价: {bucket['avg_price']['value']:.2f}")

# 范围桶聚合
result = es.search(
    index="products",
    body={
        "size": 0,
        "aggs": {
            "price_ranges": {
                "range": {
                    "field": "price",
                    "ranges": [
                        {"to": 100, "key": "低价"},
                        {"from": 100, "to": 500, "key": "中价"},
                        {"from": 500, "key": "高价"}
                    ]
                }
            }
        }
    }
)

# 直方图聚合
result = es.search(
    index="products",
    body={
        "size": 0,
        "aggs": {
            "price_histogram": {
                "histogram": {"field": "price", "interval": 100}
            }
        }
    }
)

# 日期直方图聚合
result = es.search(
    index="orders",
    body={
        "size": 0,
        "aggs": {
            "orders_per_month": {
                "date_histogram": {
                    "field": "created_at",
                    "calendar_interval": "month",
                    "format": "yyyy-MM"
                }
            }
        }
    }
)
```

## 7. 高亮显示

```python
result = es.search(
    index="articles",
    body={
        "query": {"match": {"content": "Elasticsearch"}},
        "highlight": {
            "fields": {
                "content": {
                    "pre_tags": ["<em class='highlight'>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 200,
                    "number_of_fragments": 3
                }
            }
        }
    }
)

for hit in result["hits"]["hits"]:
    if "highlight" in hit:
        for fragment in hit["highlight"]["content"]:
            print(fragment)
```
