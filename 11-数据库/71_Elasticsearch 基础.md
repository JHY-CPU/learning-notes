# Elasticsearch 基础


## 🔍 Elasticsearch 基础


Elasticsearch 概述、倒排索引原理、Index/Type/Document/Mapping、REST API（CRUD / 搜索 / 聚合）、Query DSL（term/match/bool/range）、分词器 Analyzer、聚合 Aggregation、集群架构、ELK Stack。


## Elasticsearch 概述


```
// ========== Elasticsearch ==========
// 基于 Lucene 的分布式全文搜索引擎
// 实时搜索 + 分析引擎
// REST API 接口

// ========== 核心特点 ==========
// 1. 全文搜索 (倒排索引, 分词)
// 2. 分布式 (自动分片 + 副本)
// 3. 近实时 (写入后 1s 可搜索)
// 4. Schema-less (动态 Mapping)
// 5. 聚合分析 (指标/桶/管道)

// ========== 核心概念 ==========
// ┌──────────┬────────────────────────┐
// │ ES       │ 类比 RDB               │
// ├──────────┼────────────────────────┤
// │ Index    │ Database               │
// │ Type (7.x 弃用) │ Table           │
// │ Document │ Row                    │
// │ Field    │ Column                 │
// │ Mapping  │ Schema                 │
// │ Shard    │ 分片                   │
// │ Replica  │ 副本                   │
// └──────────┴────────────────────────┘

// ========== 适用场景 ==========
// ✅ 全文搜索 (电商/文档/日志)
// ✅ 日志分析 (ELK Stack)
// ✅ 指标聚合 (监控/APM)
// ✅ 自动补全与搜索建议
// ✅ 地理空间查询

// ❌ 强一致事务
// ❌ 复杂关联查询
// ❌ 频繁更新 (ES 更新 = 删除+重建)
// ❌ 作为主数据库

// ========== 生态 ==========
// Elasticsearch + Logstash + Kibana = ELK
// Elasticsearch + Filebeat + Kibana = EFK
// APM Server / Metricbeat / Heartbeat
```


## 倒排索引


```
// ========== 倒排索引 ==========
// 核心: 从词 (Term) 到文档 (Document) 的映射

// 文档集合:
// Doc 1: "Elasticsearch is a search engine"
// Doc 2: "MongoDB is a document database"
// Doc 3: "Search engine uses inverted index"

// 倒排索引:
// ┌────────────┬──────────────────────────┐
// │ Term       │ Doc IDs                  │
// ├────────────┼──────────────────────────┤
// │ a          │ [1, 2]                   │
// │ database   │ [2]                      │
// │ document   │ [2]                      │
// │ elasticsearch │ [1]                   │
// │ engine     │ [1, 3]                   │
// │ index      │ [3]                      │
// │ inverted   │ [3]                      │
// │ is         │ [1, 2]                   │
// │ mongodb    │ [2]                      │
// │ search     │ [1, 3]                   │
// │ uses       │ [3]                      │
// └────────────┴──────────────────────────┘

// 搜索 "search engine":
// Term "search" -> [1, 3]
// Term "engine" -> [1, 3]
// 交集 -> [1, 3] (Doc 1 和 Doc 3)

// ========== 分词器 ==========
// Analyzer = Character Filter + Tokenizer + Token Filter

// Standard Analyzer (默认):
//   "Hello, World!" → [hello, world]
//   (小写 + 标点去除 + 停用词)

// 中文分词需要插件:
//   ik_smart / ik_max_word (ik 分词器)
//   "我是一个学生" → [我, 是, 一个, 学生]

// 自定义 Analyzer:
{
  "analyzer": {
    "my_analyzer": {
      "type": "custom",
      "tokenizer": "standard",
      "filter": ["lowercase", "stop", "snowball"]
    }
  }
}
```


## REST API 与 CRUD


```
// ========== ES REST API ==========
// 默认端口: 9200
// 请求体: JSON

// ========== 索引管理 ==========
// 创建索引 (带 Mapping + Settings):
PUT /products
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  },
  "mappings": {
    "properties": {
      "title": { "type": "text", "analyzer": "ik_max_word" },
      "price": { "type": "float" },
      "category": { "type": "keyword" },
      "tags": { "type": "keyword" },
      "description": { "type": "text" },
      "created_at": { "type": "date" }
    }
  }
}

// ========== CRUD ==========
// 创建文档 (自动生成 ID):
POST /products/_doc
{
  "title": "iPhone 15",
  "price": 6999,
  "category": "手机",
  "tags": ["apple", "iphone"],
  "description": "Apple 最新款智能手机"
}

// 创建/更新 (指定 ID):
PUT /products/_doc/1
{
  "title": "MacBook Pro",
  "price": 14999,
  "category": "笔记本"
}

// 查询文档:
GET /products/_doc/1

// 更新文档 (局部):
POST /products/_update/1
{
  "doc": {
    "price": 13999
  }
}

// 删除文档:
DELETE /products/_doc/1

// 删除索引:
DELETE /products

// ========== 批量操作 ==========
POST /_bulk
{ "index": { "_index": "products", "_id": 2 } }
{ "title": "iPad Air", "price": 4999 }
{ "index": { "_index": "products", "_id": 3 } }
{ "title": "AirPods", "price": 999 }
```


## Query DSL 搜索


```
// ========== Query DSL ==========
// Query Context (评分) vs Filter Context (缓存)

// ========== 叶子查询 ==========
// term: 精确匹配 (keyword)
GET /products/_search
{
  "query": {
    "term": { "category": "手机" }
  }
}

// match: 全文搜索 (分词)
GET /products/_search
{
  "query": {
    "match": { "title": "iphone手机" }
  }
}

// range: 范围查询
GET /products/_search
{
  "query": {
    "range": {
      "price": { "gte": 1000, "lte": 8000 }
    }
  }
}

// ========== 复合查询 (bool) ==========
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "手机" } }
      ],
      "filter": [
        { "term": { "category": "手机" } },
        { "range": { "price": { "gte": 1000, "lte": 8000 } } }
      ],
      "should": [
        { "match": { "description": "5G" } }    // 加分但不强制
      ],
      "must_not": [
        { "term": { "tags": "二手" } }
      ]
    }
  }
}

// ========== 分页与排序 ==========
GET /products/_search
{
  "from": 0,
  "size": 20,
  "sort": [
    { "price": { "order": "asc" } },
    "_score"
  ],
  "query": { "match_all": {} }
}

// ========== 高亮 ==========
GET /products/_search
{
  "query": { "match": { "title": "手机" } },
  "highlight": {
    "fields": {
      "title": {},
      "description": {}
    }
  }
}
```


## 聚合与集群


```
// ========== 聚合 Aggregation ==========
// Bucket (桶) + Metric (指标) + Pipeline (管道)

// 按照 category 分组, 统计数量 + 平均价格:
GET /products/_search
{
  "size": 0,
  "aggs": {
    "by_category": {
      "terms": { "field": "category" },
      "aggs": {
        "avg_price": {
          "avg": { "field": "price" }
        },
        "price_range": {
          "stats": { "field": "price" }   // count/min/max/avg/sum
        }
      }
    }
  }
}

// ========== 集群架构 ==========
// ┌────────────────────────────────────────┐
// │    Elasticsearch Cluster              │
// ├────────────────────────────────────────┤
// │  Node 1 (Master*)                     │
// │  ├── Index A Shard 0 (Primary)        │
// │  └── Index B Shard 1 (Replica)        │
// ├────────────────────────────────────────┤
// │  Node 2 (Data)                        │
// │  ├── Index A Shard 0 (Replica)        │
// │  └── Index B Shard 1 (Primary)        │
// ├────────────────────────────────────────┤
// │  Node 3 (Data + Ingest)               │
// │  ├── Index A Shard 1 (Primary)        │
// │  └── Index B Shard 0 (Replica)        │
// └────────────────────────────────────────┘

// ========== 节点类型 ==========
// Master: 集群管理 (建议 3 个, 避免脑裂)
// Data: 存储 + 搜索
// Ingest: 文档预处理管道
// Coordinating: 请求路由 (负载均衡)
// Machine Learning: 异常检测

// ========== 分片策略 ==========
// 分片过多 -> 性能下降
// 分片过少 -> 无法扩展
// 公式: 分片数 = 节点数 × 1~3
// 建议: 单分片 20-50GB

// ========== 运维 ==========
GET /_cluster/health              // 集群健康
GET /_cat/nodes?v                 // 节点列表
GET /_cat/shards?v                // 分片分布
GET /_cat/indices?v               // 索引统计
POST /_flush                      // 刷新
POST /_forcemerge?max_num_segments=1  // 合并段
```


> **Note:** 💡 ES 要点: 倒排索引实现全文搜索; text 分词 vs keyword 精确; bool 查询组合 must/filter/should/must_not; 聚合类似 GROUP BY + 统计; 分片数部署前确定不可修改; 近实时 (refresh_interval 1s); ELK 生态 (Logstash 采集 + Kibana 可视化); 不适合做主数据库。


## 练习


<!-- Converted from: 71_Elasticsearch 基础.html -->
