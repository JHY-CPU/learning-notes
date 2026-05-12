# Neo4j 图数据库概述

## 一、图数据库基础

### 1.1 属性图模型

Neo4j 使用属性图模型，核心元素：

- **节点（Node）**：实体，可带有标签（Label）和属性（Property）
- **关系（Relationship）**：连接两个节点，有方向、类型和属性
- **标签（Label）**：节点的分类标记
- **属性（Property）**：键值对，存储在节点和关系上

```
(:Person {name: "张三", age: 28})
    -[:FRIENDS_WITH {since: 2020}]->
(:Person {name: "李四", age: 30})
```

### 1.2 与关系型数据库对比

| 特性 | 关系型数据库 | 图数据库 |
|------|-------------|----------|
| 数据模型 | 表/行/列 | 节点/关系/属性 |
| 关系表示 | 外键 + JOIN | 原生关系，指针连接 |
| 多跳查询 | 多次 JOIN，性能差 | 遍历关系，性能好 |
| Schema | 严格 | 灵活 |
| 适用场景 | 结构化事务数据 | 关系密集型数据 |

---

## 二、Neo4j 架构

### 2.1 存储结构

- **节点存储**：存储节点和属性
- **关系存储**：存储关系，双向链表连接
- **属性存储**：存储键值对
- **索引存储**：标签索引、全文索引

### 2.2 ACID 事务

Neo4j 完全支持 ACID：
- 原子性：事务要么全部成功，要么全部失败
- 一致性：数据始终满足约束
- 隔离性：并发事务互不干扰
- 持久性：提交后数据永久保存

---

## 三、应用场景

- **社交网络**：用户关系、推荐好友
- **知识图谱**：实体关系、推理
- **欺诈检测**：异常关系模式
- **供应链**：追踪产品流向
- **网络安全**：攻击路径分析
- **推荐系统**：基于图的协同过滤

---

## 四、安装与连接

```bash
# Docker 安装
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687",
                               auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) AS count")
    print(result.single()["count"])

driver.close()
```

---

## 五、Python 驱动详解

### 5.1 会话与事务管理

```python
from neo4j import GraphDatabase

class Neo4jApp:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # 自动事务（推荐）
    def create_person(self, name, age):
        with self.driver.session() as session:
            result = session.execute_write(
                self._create_person_tx, name, age
            )
            return result

    @staticmethod
    def _create_person_tx(tx, name, age):
        query = "CREATE (p:Person {name: $name, age: $age}) RETURN p"
        result = tx.run(query, name=name, age=age)
        return result.single()[0]

    # 读取事务
    def find_person(self, name):
        with self.driver.session() as session:
            result = session.execute_read(
                self._find_person_tx, name
            )
            return result

    @staticmethod
    def _find_person_tx(tx, name):
        query = "MATCH (p:Person {name: $name}) RETURN p.name, p.age"
        result = tx.run(query, name=name)
        return [dict(record) for record in result]

app = Neo4jApp("bolt://localhost:7687", "neo4j", "password")
app.create_person("张三", 28)
people = app.find_person("张三")
print(people)
app.close()
```

### 5.2 批量导入

```python
import csv

def batch_import(driver, csv_path):
    with driver.session() as session:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= 1000:
                    session.execute_write(_import_batch, batch)
                    batch = []
            if batch:
                session.execute_write(_import_batch, batch)

def _import_batch(tx, batch):
    query = """
    UNWIND $rows AS row
    CREATE (p:Person {name: row.name, age: toInteger(row.age)})
    """
    tx.run(query, rows=batch)
```

---

## 六、与其他数据库对比

| 维度 | Neo4j | MongoDB | MySQL |
|------|-------|---------|-------|
| 数据模型 | 属性图 | 文档(BSON) | 关系表 |
| 查询语言 | Cypher | MQL | SQL |
| 关系处理 | 原生指针遍历 | 引用/嵌入 | JOIN |
| 多跳查询 | O(1) per hop | O(n) per hop | O(n²+) per hop |
| 事务 | 完整 ACID | 多文档事务 | 完整 ACID |
| 水平扩展 | Causal Cluster | 原生分片 | 主从/分片 |
| 最佳场景 | 关系密集 | 灵活 schema | 结构化事务 |
