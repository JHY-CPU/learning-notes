# MongoDB 聚合管道

## 1. 聚合管道概述

聚合管道（Aggregation Pipeline）是 MongoDB 最强大的数据处理工具。它将文档通过一系列**阶段（Stage）**进行处理，每个阶段对数据进行转换，上一阶段的输出作为下一阶段的输入。

```
文档集合 → $match → $group → $sort → $project → 结果
```

### 基本语法

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]
orders = db["orders"]

# 基本聚合语法
pipeline = [
    {"$match": {"status": "completed"}},
    {"$group": {"_id": "$category", "total": {"$sum": "$amount"}}},
    {"$sort": {"total": -1}}
]

results = orders.aggregate(pipeline)
for doc in results:
    print(doc)
```

## 2. $match - 过滤阶段

`$match` 用于过滤文档，语法与 `find()` 的查询条件相同。

```python
# 等值匹配
pipeline = [
    {"$match": {"status": "completed"}},
    {"$count": "total_completed"}
]
results = list(orders.aggregate(pipeline))
print(f"已完成订单数: {results[0]['total_completed']}")

# 复杂条件过滤
pipeline = [
    {"$match": {
        "status": "completed",
        "amount": {"$gte": 100},
        "created_at": {"$gte": "2024-01-01", "$lt": "2025-01-01"}
    }},
    {"$group": {
        "_id": None,
        "count": {"$sum": 1},
        "avg_amount": {"$avg": "$amount"}
    }}
]

# $match 应尽量放在管道最前面
# 原因: 先过滤减少后续阶段处理的数据量
```

## 3. $group - 分组阶段

`$group` 是聚合管道中最核心的阶段，用于分组和计算聚合值。

### 常用累加器操作符

```python
# 按类别分组，计算各类统计
pipeline = [
    {"$match": {"status": "completed"}},
    {"$group": {
        "_id": "$category",
        "total_sales": {"$sum": "$amount"},        # 求和
        "avg_price": {"$avg": "$amount"},           # 平均值
        "max_price": {"$max": "$amount"},           # 最大值
        "min_price": {"$min": "$amount"},           # 最小值
        "order_count": {"$sum": 1},                 # 计数
        "first_order": {"$min": "$created_at"},     # 最早
        "last_order": {"$max": "$created_at"}       # 最晚
    }},
    {"$sort": {"total_sales": -1}}
]

results = orders.aggregate(pipeline)
for doc in results:
    print(f"类别: {doc['_id']}, 销售额: {doc['total_sales']}")
```

### 多字段分组

```python
# 按年月分组
pipeline = [
    {"$group": {
        "_id": {
            "year": {"$year": "$created_at"},
            "month": {"$month": "$created_at"}
        },
        "revenue": {"$sum": "$amount"},
        "order_count": {"$sum": 1}
    }},
    {"$sort": {"_id.year": 1, "_id.month": 1}}
]

# 不分组（_id 为 null），全局统计
pipeline = [
    {"$group": {
        "_id": None,
        "total_revenue": {"$sum": "$amount"},
        "total_orders": {"$sum": 1},
        "avg_order_value": {"$avg": "$amount"}
    }}
]
```

### $push 和 $addToSet 收集

```python
# $push: 收集所有值到数组（可能重复）
pipeline = [
    {"$group": {
        "_id": "$category",
        "product_names": {"$push": "$product_name"},
        "prices": {"$push": "$amount"}
    }}
]

# $addToSet: 收集去重值到数组
pipeline = [
    {"$group": {
        "_id": "$user_id",
        "purchased_categories": {"$addToSet": "$category"}
    }}
]

# $first 和 $last
pipeline = [
    {"$sort": {"created_at": 1}},
    {"$group": {
        "_id": "$user_id",
        "first_purchase": {"$first": "$amount"},
        "last_purchase": {"$last": "$amount"}
    }}
]
```

## 4. $project - 投影阶段

`$project` 用于重塑文档结构，添加计算字段或筛选字段。

```python
pipeline = [
    {"$match": {"status": "completed"}},
    {"$project": {
        "order_id": 1,                # 保留字段
        "customer": "$user_name",     # 重命名
        "amount": 1,
        "discount": {"$multiply": ["$amount", 0.1]},  # 计算字段
        "year": {"$year": "$created_at"},
        "month": {"$month": "$created_at"},
        "full_address": {             # 拼接字符串
            "$concat": ["$address.province", "$address.city"]
        },
        "is_large_order": {           # 条件判断
            "$cond": {
                "if": {"$gte": ["$amount", 1000]},
                "then": True,
                "else": False
            }
        }
    }}
]

# 条件表达式
pipeline = [
    {"$project": {
        "name": 1,
        "price_tier": {
            "$switch": {
                "branches": [
                    {"case": {"$lt": ["$amount", 100]}, "then": "低价"},
                    {"case": {"$lt": ["$amount", 500]}, "then": "中价"},
                ],
                "default": "高价"
            }
        }
    }}
]
```

## 5. $sort 和 $limit / $skip

```python
# $sort 排序
pipeline = [
    {"$group": {
        "_id": "$category",
        "total": {"$sum": "$amount"}
    }},
    {"$sort": {"total": -1}},       # 降序
    {"$limit": 10}                   # 只取前10
]

# $skip + $limit 分页
page = 2
page_size = 20
pipeline = [
    {"$match": {"status": "completed"}},
    {"$sort": {"created_at": -1}},
    {"$skip": (page - 1) * page_size},
    {"$limit": page_size}
]
```

## 6. $lookup - 关联查询

`$lookup` 类似 SQL 的 JOIN，用于关联两个集合。

```python
# 基本 lookup: 订单关联用户信息
pipeline = [
    {"$match": {"status": "completed"}},
    {"$lookup": {
        "from": "users",             # 关联的集合
        "localField": "user_id",     # 当前集合的字段
        "foreignField": "_id",       # 关联集合的字段
        "as": "user_info"            # 输出数组字段名
    }},
    {"$unwind": "$user_info"},       # 展开数组
    {"$project": {
        "order_id": 1,
        "amount": 1,
        "user_name": "$user_info.name",
        "user_email": "$user_info.email"
    }}
]

# 带条件的 lookup
pipeline = [
    {"$lookup": {
        "from": "reviews",
        "let": {"product_id": "$_id"},
        "pipeline": [
            {"$match": {
                "$expr": {"$eq": ["$product_id", "$$product_id"]},
                "rating": {"$gte": 4}
            }}
        ],
        "as": "good_reviews"
    }},
    {"$addFields": {
        "good_review_count": {"$size": "$good_reviews"}
    }}
]
```

## 7. $unwind - 数组展开

`$unwind` 将数组字段展开为多个文档，每个元素生成一个文档。

```python
# 基本展开
pipeline = [
    {"$unwind": "$tags"},            # tags 数组展开
    {"$group": {
        "_id": "$tags",
        "count": {"$sum": 1}
    }},
    {"$sort": {"count": -1}},
    {"$limit": 10}
]

# 处理空数组和缺失字段
pipeline = [
    {"$unwind": {
        "path": "$tags",
        "preserveNullAndEmptyArrays": True  # 保留没有数组或空数组的文档
    }}
]

# 带数组索引的展开
pipeline = [
    {"$unwind": {
        "path": "$items",
        "includeArrayIndex": "item_index"   # 添加索引字段
    }}
]
```

## 8. $addFields / $set - 添加计算字段

```python
pipeline = [
    {"$addFields": {
        "total_with_tax": {"$multiply": ["$amount", 1.13]},
        "order_year": {"$year": "$created_at"},
        "full_name": {"$concat": ["$first_name", " ", "$last_name"]},
        "item_count": {"$size": {"$ifNull": ["$items", []]}}
    }}
]
```

## 9. 管道优化技巧

### 优化原则

```python
# 原则1: $match 尽量放在最前面
# 优化前
pipeline_bad = [
    {"$group": {"_id": "$category", "total": {"$sum": "$amount"}}},
    {"$match": {"total": {"$gt": 1000}}}  # 无法使用索引
]

# 优化后: 先过滤再分组
pipeline_good = [
    {"$match": {"status": "completed"}},   # 可使用索引
    {"$group": {"_id": "$category", "total": {"$sum": "$amount"}}}
]

# 原则2: $project 尽早减少字段
pipeline = [
    {"$match": {"status": "completed"}},
    {"$project": {"category": 1, "amount": 1}},  # 只保留需要的字段
    {"$group": {"_id": "$category", "total": {"$sum": "$amount"}}}
]

# 原则3: 使用 allowDiskUse 处理大数据集
results = orders.aggregate(pipeline, allowDiskUse=True)
```

### 使用 explain 分析管道

```python
# 查看聚合管道的执行计划
explain_result = db.command(
    "aggregate", "orders",
    pipeline=pipeline,
    explain=True
)
print(explain_result)
```

## 10. 实战案例

### 案例1: 月度销售报表

```python
pipeline = [
    {"$match": {
        "status": "completed",
        "created_at": {"$gte": "2024-01-01"}
    }},
    {"$group": {
        "_id": {
            "year": {"$year": "$created_at"},
            "month": {"$month": "$created_at"}
        },
        "revenue": {"$sum": "$amount"},
        "order_count": {"$sum": 1},
        "avg_order_value": {"$avg": "$amount"},
        "unique_customers": {"$addToSet": "$user_id"}
    }},
    {"$addFields": {
        "customer_count": {"$size": "$unique_customers"}
    }},
    {"$project": {
        "unique_customers": 0  # 移除不需要的数组字段
    }},
    {"$sort": {"_id.year": 1, "_id.month": 1}}
]

for report in orders.aggregate(pipeline):
    period = report["_id"]
    print(f"{period['year']}年{period['month']}月:")
    print(f"  收入: {report['revenue']:.2f}")
    print(f"  订单数: {report['order_count']}")
    print(f"  客户数: {report['customer_count']}")
```

### 案例2: 用户购买行为分析

```python
pipeline = [
    {"$match": {"status": "completed"}},
    {"$group": {
        "_id": "$user_id",
        "total_spent": {"$sum": "$amount"},
        "order_count": {"$sum": 1},
        "first_order": {"$min": "$created_at"},
        "last_order": {"$max": "$created_at"},
        "categories": {"$addToSet": "$category"}
    }},
    {"$addFields": {
        "avg_order_value": {"$round": [{"$divide": ["$total_spent", "$order_count"]}, 2]},
        "category_count": {"$size": "$categories"},
        "customer_tier": {
            "$switch": {
                "branches": [
                    {"case": {"$gte": ["$total_spent", 10000]}, "then": "铂金"},
                    {"case": {"$gte": ["$total_spent", 5000]}, "then": "黄金"},
                    {"case": {"$gte": ["$total_spent", 1000]}, "then": "白银"},
                ],
                "default": "普通"
            }
        }
    }},
    {"$sort": {"total_spent": -1}},
    {"$limit": 100}
]
```

### 案例3: 商品关联分析

```python
pipeline = [
    {"$match": {"status": "completed"}},
    {"$group": {
        "_id": "$order_id",
        "products": {"$push": "$product_name"}
    }},
    {"$match": {"products.1": {"$exists": True}}},  # 至少包含2个商品
    {"$unwind": "$products"},
    {"$unwind": "$products"},  # 自关联需要 $lookup
]
```
