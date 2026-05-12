# MongoDB 事务与变更流

## 1. 多文档事务

MongoDB 4.0 引入了副本集上的多文档事务，4.2 扩展到分片集群。事务保证一组操作的原子性——要么全部成功，要么全部回滚。

### 基本事务用法

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["banking"]
accounts = db["accounts"]
transactions_col = db["transactions"]

# 转账操作：A扣款 + B加款 + 记录交易
def transfer_money(session, from_id, to_id, amount):
    accounts = session.client.banking.accounts
    transactions = session.client.banking.transactions

    # 检查余额
    sender = accounts.find_one({"_id": from_id}, session=session)
    if sender["balance"] < amount:
        raise ValueError("余额不足")

    # 扣款
    accounts.update_one(
        {"_id": from_id},
        {"$inc": {"balance": -amount}},
        session=session
    )

    # 加款
    accounts.update_one(
        {"_id": to_id},
        {"$inc": {"balance": amount}},
        session=session
    )

    # 记录交易
    from datetime import datetime
    transactions.insert_one({
        "from": from_id,
        "to": to_id,
        "amount": amount,
        "timestamp": datetime.utcnow(),
        "status": "completed"
    }, session=session)

# 使用事务
with client.start_session() as session:
    try:
        session.with_transaction(
            lambda s: transfer_money(s, "account_a", "account_b", 100)
        )
        print("转账成功")
    except Exception as e:
        print(f"转账失败: {e}")
        # 事务自动回滚
```

### 手动事务控制

```python
with client.start_session() as session:
    # 开始事务
    session.start_transaction(
        read_concern=ReadConcern("snapshot"),
        write_concern=WriteConcern(w="majority"),
        read_preference=ReadPreference.PRIMARY
    )

    try:
        # 在事务中执行操作
        accounts.update_one(
            {"_id": "account_a"},
            {"$inc": {"balance": -50}},
            session=session
        )

        accounts.update_one(
            {"_id": "account_b"},
            {"$inc": {"balance": 50}},
            session=session
        )

        # 提交事务
        session.commit_transaction()
        print("事务提交成功")

    except Exception as e:
        # 回滚事务
        session.abort_transaction()
        print(f"事务回滚: {e}")
```

### 事务的限制

```python
"""
事务的限制和注意事项:

1. 超时限制: 事务默认超时 60 秒
2. Oplog 大小限制: 单个事务的 Oplog 条目不能超过 16MB
3. 系统集合: 不能在事务中操作 config/admin/local 数据库
4. 集合操作: 不能在事务中创建集合/索引（某些版本限制）
5. 特殊命令: 不能在事务中使用 killCursors, getMore 等命令
6. 分片集群: 需要 MongoDB 4.2+

性能建议:
- 事务尽量短小，减少锁持有时间
- 避免在事务中进行大量读写
- 使用读关注 "snapshot" 获得一致性快照
- 使用写关注 "majority" 确保持久性
"""
```

## 2. Change Streams（变更流）

Change Streams 允许应用程序实时监听集合、数据库或整个集群的变更事件，而不需要轮询。

### 监听单个集合

```python
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]

# 监听 orders 集合的变更
with db.orders.watch() as stream:
    print("开始监听 orders 集合变更...")
    for change in stream:
        print(f"操作类型: {change['operationType']}")
        print(f"文档ID: {change['documentKey']['_id']}")
        if change['operationType'] == 'insert':
            print(f"新文档: {change['fullDocument']}")
        elif change['operationType'] == 'update':
            print(f"更新字段: {change['updateDescription']['updatedFields']}")
            print(f"移除字段: {change['updateDescription']['removedFields']}")
        print("---")
```

### 使用 Resume Token 恢复

Resume Token 允许从断点处继续监听变更，确保不丢失事件。

```python
import json
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]

def save_resume_token(token):
    """保存 Resume Token 到文件"""
    with open("resume_token.json", "w") as f:
        json.dump({"resume_token": str(token)}, f)

def load_resume_token():
    """加载 Resume Token"""
    try:
        with open("resume_token.json", "r") as f:
            data = json.load(f)
            return data.get("resume_token")
    except FileNotFoundError:
        return None

# 从上次断点恢复监听
resume_token = load_resume_token()
pipeline = []  # 可选的过滤管道

if resume_token:
    print(f"从断点恢复: {resume_token}")
    with db.orders.watch(pipeline, resume_after=resume_token) as stream:
        for change in stream:
            print(f"变更: {change['operationType']}")
            # 持续保存最新的 Resume Token
            save_resume_token(stream.resume_token)
else:
    print("从当前位置开始监听")
    with db.orders.watch(pipeline) as stream:
        for change in stream:
            print(f"变更: {change['operationType']}")
            save_resume_token(stream.resume_token)
```

### 过滤变更事件

```python
# 使用聚合管道过滤特定变更
pipeline = [
    # 只监听插入和更新操作
    {"$match": {
        "operationType": {"$in": ["insert", "update"]}
    }},
    # 只监听特定字段的变更
    {"$match": {
        "updateDescription.updatedFields.status": {"$exists": True}
    }},
    # 只返回特定字段
    {"$project": {
        "operationType": 1,
        "documentKey": 1,
        "fullDocument": 1,
        "updateDescription": 1
    }}
]

with db.orders.watch(pipeline) as stream:
    for change in stream:
        if change['operationType'] == 'update':
            new_status = change['updateDescription']['updatedFields'].get('status')
            print(f"订单 {change['documentKey']['_id']} 状态变为: {new_status}")
```

### 监听数据库级别变更

```python
# 监听整个数据库的变更
with client.ecommerce.watch() as stream:
    for change in stream:
        print(f"集合: {change['ns']['coll']}")
        print(f"操作: {change['operationType']}")
```

### 监听集群级别变更

```python
# 监听整个集群的变更（需要 admin 数据库权限）
with client.watch() as stream:
    for change in stream:
        db_name = change['ns']['db']
        coll_name = change['ns']['coll']
        print(f"[{db_name}.{coll_name}] {change['operationType']}")
```

## 3. Change Streams 实战案例

### 案例1: 实时数据同步

```python
from pymongo import MongoClient
import threading

def sync_to_elasticsearch(es_client, change):
    """将 MongoDB 变更同步到 Elasticsearch"""
    op_type = change['operationType']
    doc_id = str(change['documentKey']['_id'])

    if op_type == 'insert':
        es_client.index(
            index="orders",
            id=doc_id,
            body=change['fullDocument']
        )
    elif op_type == 'update':
        es_client.update(
            index="orders",
            id=doc_id,
            body={"doc": change['updateDescription']['updatedFields']}
        )
    elif op_type == 'delete':
        es_client.delete(index="orders", id=doc_id)

def watch_and_sync(db):
    """监听变更并同步"""
    with db.orders.watch() as stream:
        for change in stream:
            try:
                sync_to_elasticsearch(es_client, change)
            except Exception as e:
                print(f"同步失败: {e}")

# 在后台线程中运行
thread = threading.Thread(target=watch_and_sync, args=(db,), daemon=True)
thread.start()
```

### 案例2: 实时通知系统

```python
from pymongo import MongoClient
import smtplib

def send_notification(order):
    """发送邮件通知"""
    print(f"发送通知: 订单 {order['_id']} 已完成")
    # 实际代码: 发送邮件/短信/推送通知

pipeline = [
    {"$match": {
        "operationType": "update",
        "updateDescription.updatedFields.status": "completed"
    }}
]

with db.orders.watch(pipeline) as stream:
    for change in stream:
        order = change['fullDocument']
        if order and order.get('notify_user'):
            send_notification(order)
```

### 案例3: 审计日志

```python
from pymongo import MongoClient
from datetime import datetime

def create_audit_log(change):
    """创建审计日志"""
    audit_entry = {
        "timestamp": datetime.utcnow(),
        "collection": change['ns']['coll'],
        "operation": change['operationType'],
        "document_id": change['documentKey']['_id'],
        "details": {}
    }

    if change['operationType'] == 'insert':
        audit_entry['details']['new_document'] = change['fullDocument']
    elif change['operationType'] == 'update':
        audit_entry['details']['changes'] = change['updateDescription']
    elif change['operationType'] == 'delete':
        audit_entry['details']['deleted_document'] = change.get('fullDocument')

    return audit_entry

# 监听所有集合的变更并记录审计日志
audit_collection = db["audit_logs"]

with client.ecommerce.watch() as stream:
    for change in stream:
        # 跳过审计日志集合本身
        if change['ns']['coll'] == 'audit_logs':
            continue

        audit_entry = create_audit_log(change)
        audit_collection.insert_one(audit_entry)
        print(f"审计: {audit_entry['operation']} on {audit_entry['collection']}")
```

## 4. Change Streams 配置选项

```python
# 完整的配置示例
with db.orders.watch(
    pipeline=[
        {"$match": {"operationType": {"$in": ["insert", "update", "delete"]}}}
    ],
    full_document="updateLookup",       # 更新时返回完整文档
    full_document_before_change="off",  # 不返回变更前的文档
    max_await_time_ms=10000,            # 最大等待时间
    batch_size=100,                     # 每批返回的变更数量
    collation={"locale": "en"}          # 排序规则
) as stream:
    for change in stream:
        pass
```

### full_document 选项说明

| 选项 | 说明 |
|------|------|
| `default` | insert 返回完整文档，update 只返回变更字段 |
| `updateLookup` | update 时返回更新后的完整文档 |
| `whenAvailable` | 尽可能返回完整文档 |
| `required` | 必须返回完整文档，否则报错 |
