# MongoDB CRUD 操作

## 1. 插入操作

### insertOne - 插入单条文档

```python
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]
users = db["users"]

# 插入单条文档
doc = {
    "name": "李四",
    "email": "lisi@example.com",
    "age": 25,
    "status": "active",
    "tags": ["vip", "new_user"],
    "address": {
        "province": "广东省",
        "city": "深圳市"
    },
    "created_at": datetime.utcnow()
}
result = users.insert_one(doc)
print(f"插入成功，ID: {result.inserted_id}")
# 输出: 插入成功，ID: 64a1b2c3d4e5f6a7b8c9d0e1
```

### insertMany - 批量插入

```python
docs = [
    {"name": "王五", "age": 30, "status": "active"},
    {"name": "赵六", "age": 22, "status": "inactive"},
    {"name": "孙七", "age": 35, "status": "active"},
    {"name": "周八", "age": 28, "status": "active"},
]

result = users.insert_many(docs)
print(f"插入 {len(result.inserted_ids)} 条文档")
# ordered=False 表示即使某条失败也继续插入其余
result = users.insert_many(docs, ordered=False)
```

## 2. 查询操作

### find - 基本查询

```python
# 查询所有活跃用户
cursor = users.find({"status": "active"})
for user in cursor:
    print(user["name"], user["age"])

# 查询单条
user = users.find_one({"name": "张三"})
print(user)

# 条件查询: 年龄大于25且状态为active
cursor = users.find({
    "age": {"$gt": 25},
    "status": "active"
})

# 嵌套文档查询
cursor = users.find({"address.city": "深圳"})
```

### 比较操作符

```python
# $gt 大于, $gte 大于等于, $lt 小于, $lte 小于等于
users.find({"age": {"$gt": 25}})
users.find({"age": {"$gte": 18, "$lte": 60}})

# $ne 不等于
users.find({"status": {"$ne": "deleted"}})

# $in 在数组中, $nin 不在数组中
users.find({"age": {"$in": [25, 30, 35]}})
users.find({"status": {"$nin": ["banned", "deleted"]}})

# $exists 字段是否存在
users.find({"email": {"$exists": True}})
```

### 逻辑操作符

```python
# $and（默认隐式and）
users.find({"age": {"$gt": 20}, "status": "active"})

# $or
users.find({
    "$or": [
        {"age": {"$lt": 20}},
        {"status": "vip"}
    ]
})

# $not
users.find({"age": {"$not": {"$gte": 18}}})

# $nor（都不满足）
users.find({
    "$nor": [
        {"status": "banned"},
        {"age": {"$lt": 18}}
    ]
})
```

### 投影（Projection）

```python
# 只返回 name 和 age 字段，排除 _id
cursor = users.find(
    {"status": "active"},
    {"name": 1, "age": 1, "_id": 0}
)
for user in cursor:
    print(user)
# 输出: {'name': '张三', 'age': 28}

# 排除某些字段
cursor = users.find({}, {"password": 0, "salt": 0})
```

### 排序与分页

```python
# sort: 1 升序, -1 降序
cursor = users.find({"status": "active"}).sort("age", -1)

# 多字段排序
cursor = users.find().sort([("status", 1), ("age", -1)])

# skip + limit 分页
page = 2
page_size = 10
cursor = users.find().skip((page - 1) * page_size).limit(page_size)

# 组合使用
cursor = (users
    .find({"status": "active"})
    .sort("created_at", -1)
    .skip(0)
    .limit(5))
```

## 3. 更新操作

### updateOne - 更新单条

```python
# $set 设置字段值
result = users.update_one(
    {"name": "张三"},
    {"$set": {"age": 29, "status": "vip"}}
)
print(f"匹配: {result.matched_count}, 修改: {result.modified_count}")

# $unset 删除字段
users.update_one({"name": "张三"}, {"$unset": {"temp_field": ""}})

# $inc 递增
users.update_one({"name": "张三"}, {"$inc": {"login_count": 1}})

# $rename 重命名字段
users.update_one({"name": "张三"}, {"$rename": {"old_name": "new_name"}})

# upsert: 不存在则插入
users.update_one(
    {"name": "新用户"},
    {"$set": {"age": 20, "status": "new"}},
    upsert=True
)
```

### 数组更新操作符

```python
# $push 添加到数组
users.update_one(
    {"name": "张三"},
    {"$push": {"tags": "premium"}}
)

# $addToSet 添加到数组（去重）
users.update_one(
    {"name": "张三"},
    {"$addToSet": {"tags": "active"}}
)

# $pull 从数组中移除
users.update_one(
    {"name": "张三"},
    {"$pull": {"tags": "inactive"}}
)

# $pop 移除首/尾元素
users.update_one({"name": "张三"}, {"$pop": {"tags": 1}})   # 移除最后一个
users.update_one({"name": "张三"}, {"$pop": {"tags": -1}})  # 移除第一个

# $each 配合 $push/$addToSet 批量操作
users.update_one(
    {"name": "张三"},
    {"$addToSet": {"tags": {"$each": ["tag1", "tag2", "tag3"]}}}
)

# $pullAll 移除多个值
users.update_one(
    {"name": "张三"},
    {"$pullAll": {"tags": ["tag1", "tag2"]}}
)
```

### updateMany - 批量更新

```python
# 批量更新所有非活跃用户
result = users.update_many(
    {"status": "inactive"},
    {"$set": {"status": "archived", "archived_at": datetime.utcnow()}}
)
print(f"更新了 {result.modified_count} 条文档")
```

### replaceOne - 替换整个文档

```python
# 替换文档（保留 _id）
users.replace_one(
    {"name": "张三"},
    {"name": "张三", "age": 29, "status": "vip", "version": 2}
)
```

## 4. 删除操作

```python
# deleteOne 删除单条
result = users.delete_one({"name": "测试用户"})
print(f"删除了 {result.deleted_count} 条文档")

# deleteMany 批量删除
result = users.delete_many({"status": "deleted"})
print(f"删除了 {result.deleted_count} 条文档")

# 删除集合中所有文档（集合结构保留）
users.delete_many({})

# drop_collection 删除整个集合
db.drop_collection("temp_collection")
```

## 5. 批量操作（Bulk Write）

`bulk_write` 允许在一次请求中执行多种操作，减少网络往返。

```python
from pymongo import InsertOne, UpdateOne, DeleteOne, ReplaceOne

operations = [
    # 插入新用户
    InsertOne({"name": "批量用户1", "age": 20, "status": "active"}),
    InsertOne({"name": "批量用户2", "age": 25, "status": "active"}),

    # 更新已有用户
    UpdateOne(
        {"name": "张三"},
        {"$set": {"age": 30}},
        upsert=False
    ),

    # 删除用户
    DeleteOne({"name": "待删除用户"}),

    # 替换文档
    ReplaceOne(
        {"name": "李四"},
        {"name": "李四", "age": 26, "status": "vip"}
    ),
]

result = users.bulk_write(operations, ordered=True)
print(f"插入: {result.inserted_count}")
print(f"修改: {result.modified_count}")
print(f"删除: {result.deleted_count}")
```

### 无序批量操作

```python
# ordered=False: 某个操作失败不影响其余操作执行
operations = [
    InsertOne({"name": f"用户{i}", "age": i})
    for i in range(1000)
]
result = users.bulk_write(operations, ordered=False)
print(f"批量插入了 {result.inserted_count} 条文档")
```

## 6. 实用查询技巧

### 正则表达式查询

```python
import re

# 名字以"张"开头的用户
cursor = users.find({"name": re.compile(r"^张")})

# 包含"三"的用户名（不区分大小写）
cursor = users.find({"name": re.compile(r"三", re.IGNORECASE)})
```

### $where 高级查询

```python
# 使用 JavaScript 表达式（性能较低，优先用操作符）
cursor = users.find({"$where": "this.age > this.avg_score"})
```

### 计数与去重

```python
# 计数
count = users.count_documents({"status": "active"})
print(f"活跃用户数: {count}")

# 去重
cities = users.distinct("address.city")
print(f"所有城市: {cities}")
```
