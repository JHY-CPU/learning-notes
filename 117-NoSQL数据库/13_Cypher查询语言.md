# Cypher 查询语言

## 一、Cypher 设计哲学与底层原理

### 1.1 声明式图查询

Cypher 是一种声明式图查询语言（由 Neo4j 于 2011 年发明，2015 年开放为 openCypher 标准）。与 SQL 类似，用户描述「要什么」而非「怎么做」。

**核心语法：ASCII-Art 模式匹配**

```
(节点)-[关系]->(节点)
```

- `(node)` — 圆括号表示节点（顶点）
- `-[rel]->` — 方括号+箭头表示有向关系
- `*1..3` — 可变长度路径
- `{key: value}` — 节点/关系属性

### 1.2 底层执行机制

Cypher 查询经过以下流水线：

```
Cypher 文本 → Parser (ANTLR4) → AST → Logical Plan → Physical Plan → Execution
```

- **Parser**：基于 ANTLR4 语法解析器生成器，将 Cypher 解析为 AST
- **Logical Plan**：关系代数表达式（Expand、Filter、Projection 等算子）
- **Physical Plan**：决定具体的执行策略（如 Expand Into vs Expand All、使用哪个索引）
- **Execution**：Neo4j 使用 Volcano 模型迭代器模式执行，每个算子实现 `next()` 方法

**关键概念：Expand 算子**

Expand 是图查询的核心算子，分为三种变体：

| 算子 | 起点已知 | 终点已知 | 实现方式 |
|------|---------|---------|---------|
| **Expand(All)** | 是 | 否 | 从起点出发，遍历所有关系 |
| **Expand(Into)** | 是 | 是 | 检查两个已知节点间是否存在指定关系 |
| **Expand(Optional)** | 是 | 否 | 左外连接语义，即使无关系也保留起点 |

**Neo4j 的存储访问**：Neo4j 使用邻接表（Adjacency List）存储关系。每个节点在存储中有一个指向其第一条关系的指针，关系通过双向链表连接（`first_rel → next_rel → ...`）。Expand 操作沿着这条链表遍历，时间复杂度为 O(关系数)，与图总大小无关——这是图数据库的核心优势。

---

## 二、节点与关系 CRUD

### 2.1 创建节点

```cypher
// 创建单个节点
CREATE (p:Person {name: "张三", age: 28, city: "北京", created: datetime()})

// 创建多个节点（一个 CREATE 可以创建多条）
CREATE 
  (p1:Person {name: "李四", age: 30}),
  (p2:Person {name: "王五", age: 25}),
  (c:Company {name: "腾讯", founded: date("1998-11-11")}),
  (skill:Skill {name: "Python"}),
  (skill2:Skill {name: "Java"})

// 创建节点和关系（整个模式作为一个 CREATE）
CREATE (p:Person {name: "张三"})-[:WORKS_AT {since: date("2020-06-01"), role: "高级工程师"}]->(c:Company {name: "阿里巴巴"})

// 批量创建（使用 UNWIND + 参数）
:param employees => [
  {name: "赵六", age: 35, company: "字节跳动"},
  {name: "孙七", age: 28, company: "美团"},
  {name: "周八", age: 32, company: "字节跳动"}
]

UNWIND $employees AS emp
MERGE (c:Company {name: emp.company})
MERGE (p:Person {name: emp.name})
SET p.age = emp.age
MERGE (p)-[:WORKS_AT]->(c)
```

### 2.2 查询进阶

```cypher
// 基础查询
MATCH (p:Person {name: "张三"}) RETURN p

// WHERE 条件（支持丰富的表达式）
MATCH (p:Person)
WHERE p.age > 25 
  AND p.city = "北京"
  AND p.name STARTS WITH "张"
  AND p.created > datetime("2024-01-01")
RETURN p.name, p.age

// 正则匹配
MATCH (p:Person)
WHERE p.email =~ ".*@company\\.com$"
RETURN p.name, p.email

// IN 操作符
MATCH (p:Person)
WHERE p.city IN ["北京", "上海", "广州", "深圳"]
RETURN p.name, p.city

// IS NOT NULL / IS NULL
MATCH (p:Person)
WHERE p.email IS NOT NULL
RETURN p.name, p.email

// 列表/范围查询
MATCH (p:Person)
WHERE p.age IN range(25, 35)
RETURN p.name, p.age

// 查询关系及其属性
MATCH (p:Person)-[r:FRIENDS_WITH]->(friend:Person)
WHERE p.name = "张三" AND r.since > date("2020-01-01")
RETURN friend.name, r.since, duration.between(r.since, date()).years AS years
```

### 2.3 更新操作

```cypher
// SET 更新/添加属性
MATCH (p:Person {name: "张三"})
SET p.age = 29, 
    p.city = "上海",
    p.updated = datetime()
RETURN p

// SET 添加标签
MATCH (p:Person {name: "张三"})
SET p:Engineer:Senior

// REMOVE 删除属性
MATCH (p:Person {name: "张三"})
REMOVE p.city

// REMOVE 删除标签
MATCH (p:Person {name: "张三"})
REMOVE p:Engineer

// 删除节点（必须先删除关联的关系）
MATCH (p:Person {name: "张三"})-[r]-()
DELETE r, p

// DETACH DELETE 同时删除节点和所有关系
MATCH (p:Person {name: "张三"})
DETACH DELETE p
```

### 2.4 MERGE 深入解析

`MERGE` = MATCH + CREATE 的组合。如果模式存在则匹配，不存在则创建。

```cypher
// MERGE 节点
MERGE (p:Person {name: "张三"})
ON CREATE SET p.created = datetime(), p.source = "api"
ON MATCH SET p.lastSeen = datetime()
RETURN p

// MERGE 关系（必须确保两个端点都已存在）
MATCH (p:Person {name: "张三"})
MATCH (c:Company {name: "腾讯"})
MERGE (p)-[r:WORKS_AT]->(c)
ON CREATE SET r.since = date(), r.role = "新人"
RETURN p.name, c.name, r

// 常见错误：MERGE 包含属性的完整模式时，所有属性必须完全匹配
MERGE (p:Person {name: "张三", age: 28})  // 如果 name="张三" 但 age!=28，会创建新节点！

// 正确做法：先 MERGE 关键属性，再 SET 其他属性
MERGE (p:Person {name: "张三"})
SET p.age = 28, p.city = "北京"
```

**常见坑点**：
- `MERGE` 的属性必须是**精确匹配**。`MERGE (p:Person {name: "张三", age: 28})` 会检查是否存在 name="张三" **且** age=28 的节点，如果只有 name 匹配但 age 不同，会创建一个新节点
- `MERGE` 在并发环境下使用**悲观锁**，可能导致死锁。解决办法：对同一批节点按确定顺序 MERGE

---

## 三、模式匹配深度

### 3.1 可变长度路径

```cypher
// 固定长度路径（3跳）
MATCH (p:Person {name: "张三"})-[:FRIENDS_WITH*3]-(friend)
RETURN DISTINCT friend.name

// 范围长度路径（1-3跳）
MATCH (p:Person {name: "张三"})-[:FRIENDS_WITH*1..3]-(friend)
WHERE friend <> p
RETURN DISTINCT friend.name

// 任意长度路径（最多 5 跳，防止无限循环）
MATCH (p:Person {name: "张三"})-[:FRIENDS_WITH*..5]-(friend)
WHERE friend <> p
RETURN DISTINCT friend.name

// 保留路径中的中间节点
MATCH path = (p:Person {name: "张三"})-[:FRIENDS_WITH*1..3]-(friend)
WHERE friend <> p
RETURN [n IN nodes(path) | n.name] AS path_names,
       length(path) AS hops
ORDER BY hops
```

### 3.2 最短路径

```cypher
// 单条最短路径（无权重）
MATCH (start:Person {name: "张三"}), (end:Person {name: "王五"})
MATCH path = shortestPath((start)-[:FRIENDS_WITH*]-(end))
RETURN path, length(path) AS hops

// 最短路径 + 条件过滤（关系类型、方向等）
MATCH path = shortestPath(
  (start:Person {name: "张三"})-[:FRIENDS_WITH|COLLEAGUE*]-(end:Person {name: "王五"})
)
WHERE start <> end
RETURN path

// 所有最短路径
MATCH (start:Person {name: "张三"}), (end:Person {name: "王五"})
MATCH path = allShortestPaths((start)-[:FRIENDS_WITH*]-(end))
RETURN path, length(path) AS hops
```

### 3.3 OPTIONAL MATCH（左外连接）

```cypher
// 即使没有匹配的关系，也保留左端节点
MATCH (p:Person {name: "张三"})
OPTIONAL MATCH (p)-[:HAS_HOBBY]->(h:Hobby)
RETURN p.name, collect(h.name) AS hobbies
// 如果张三没有爱好，hobbies 返回空列表 []

// OPTIONAL MATCH 用于多关系查询
MATCH (p:Person)
WHERE p.city = "北京"
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
RETURN p.name, 
       c.name AS company,
       collect(DISTINCT s.name) AS skills
```

### 3.4 WHERE 中的关系模式

```cypher
// EXISTS：检查是否存在某种关系模式
MATCH (p:Person)
WHERE EXISTS((p)-[:WORKS_AT]->(:Company {name: "腾讯"}))
RETURN p.name

// NOT EXISTS：检查不存在
MATCH (p:Person)
WHERE NOT EXISTS((p)-[:FRIENDS_WITH]-())
RETURN p.name AS 孤立用户

// 模式作为谓词
MATCH (p:Person)
WHERE EXISTS {
  MATCH (p)-[:HAS_SKILL]->(s:Skill)
  WHERE s.name = "Python"
}
RETURN p.name
```

---

## 四、聚合与集合操作

### 4.1 聚合函数

```cypher
// 基础聚合
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name,
       count(p) AS employee_count,
       avg(p.age) AS avg_age,
       min(p.age) AS min_age,
       max(p.age) AS max_age,
       sum(p.salary) AS total_salary
ORDER BY employee_count DESC

// COUNT 区别
// count(n) — 计数非空值
// count(*) — 计数所有行（包括 null）
MATCH (p:Person)
RETURN count(p.email), count(*) 
// 如果某些人没有 email，count(p.email) < count(*)

// collect() — 聚合为列表
MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: "腾讯"})
RETURN c.name,
       collect(p.name) AS employees,
       collect(DISTINCT p.city) AS cities

// percentileCont / percentileDisc — 百分位数
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name,
       percentileCont(p.salary, 0.5) AS median_salary,
       percentileCont(p.salary, 0.95) AS p95_salary
```

### 4.2 WITH 子句（管道）

`WITH` 是 Cypher 的"管道"，将前一步结果传递给后一步。它在逻辑上将查询分为多个阶段。

```cypher
// 过滤聚合结果
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WITH c, count(p) AS cnt
WHERE cnt > 100
RETURN c.name, cnt
ORDER BY cnt DESC

// 中间排序 + 限制
MATCH (p:Person)
WITH p
ORDER BY p.age DESC
LIMIT 10
MATCH (p)-[:HAS_SKILL]->(s:Skill)
RETURN p.name, collect(s.name) AS skills

// 在 WITH 中做计算
MATCH (p:Person)-[:BOUGHT]->(product:Product)
WITH p, 
     count(product) AS buy_count,
     sum(product.price) AS total_spent
WHERE total_spent > 10000
RETURN p.name, buy_count, total_spent
ORDER BY total_spent DESC
```

### 4.3 UNWIND — 列表展开

```cypher
// 将列表展开为多行
WITH ["Python", "Java", "Go", "Rust"] AS langs
UNWIND langs AS lang
MERGE (s:Skill {name: lang})
RETURN s

// 实际场景：批量导入 JSON 数据
:param data => [
  {name: "张三", skills: ["Python", "SQL"]},
  {name: "李四", skills: ["Java", "Go"]},
  {name: "王五", skills: ["Python", "Rust", "Go"]}
]

UNWIND $data AS person
MERGE (p:Person {name: person.name})
WITH p, person.skills AS skills
UNWIND skills AS skill_name
MERGE (s:Skill {name: skill_name})
MERGE (p)-[:HAS_SKILL]->(s)
RETURN p.name, collect(skill_name) AS skills

// 展开 + 聚合组合
WITH [[1, 2], [3, 4], [5, 6]] AS lists
UNWIND lists AS list
UNWIND list AS item
RETURN sum(item) AS total  // = 21
```

### 4.4 CALL 子查询

```cypher
// 使用 CALL 对每个子集执行子查询
MATCH (c:Company)
WITH c
ORDER BY c.name
LIMIT 5
CALL {
  WITH c
  MATCH (p:Person)-[:WORKS_AT]->(c)
  RETURN count(p) AS emp_count, avg(p.age) AS avg_age
}
RETURN c.name, emp_count, avg_age

// 有条件执行
MATCH (p:Person)
CALL {
  WITH p
  OPTIONAL MATCH (p)-[:HAS_HOBBY]->(h:Hobby)
  RETURN collect(h.name) AS hobbies
}
WHERE size(hobbies) > 2
RETURN p.name, hobbies
```

---

## 五、图遍历与路径查询

### 5.1 二度关系分析

```cypher
// 二度好友（排除自己和一度好友）
MATCH (me:Person {name: "张三"})-[:FRIENDS_WITH*2]-(fof:Person)
WHERE fof <> me
  AND NOT (me)-[:FRIENDS_WITH]-(fof)
RETURN DISTINCT fof.name, 
       count(*) AS common_friends
ORDER BY common_friends DESC

// 共同好友
MATCH (p1:Person {name: "张三"})-[:FRIENDS_WITH]-(mutual:Person)-[:FRIENDS_WITH]-(p2:Person {name: "李四"})
RETURN collect(mutual.name) AS mutual_friends

// 二度关系 + 关系类型混合
MATCH (me:Person {name: "张三"})-[:FRIENDS_WITH|COLLEAGUE*2]-(connection)
WHERE connection <> me
  AND NOT (me)-[:FRIENDS_WITH|COLLEAGUE]-(connection)
RETURN DISTINCT connection.name
```

### 5.2 图遍历性能分析

```cypher
// 查看执行计划
EXPLAIN MATCH (p:Person {name: "张三"})-[:FRIENDS_WITH*1..3]-(f)
RETURN DISTINCT f.name

// 实际执行分析
PROFILE MATCH (p:Person {name: "张三"})-[:FRIENDS_WITH*1..3]-(f)
RETURN DISTINCT f.name
```

**性能关键点**：
- 图遍历的时间复杂度是 O(b^d)，其中 b 是平均出度，d 是深度
- 假设平均出度 = 50，3 跳遍历 = 50^3 = 125,000 次关系访问
- 实际中 Neo4j 会在 Expand 阶段去重，避免重复访问同一节点
- **5 跳以上的无权重遍历在大型图上通常不可行**，必须加条件过滤

### 5.3 APOC 库的路径查询

```cypher
// 安装 APOC 后可使用更强大的路径算法
CALL apoc.path.expand(
  startNode,           // 起点节点
  "FRIENDS_WITH|COLLEAGUE",  // 关系类型
  "",                  // 终点标签
  1,                   // 最小深度
  4                    // 最大深度
) YIELD path
RETURN path

// 带过滤的遍历
CALL apoc.path.expandConfig(
  startNode,
  {
    relationshipFilter: "FRIENDS_WITH>",
    labelFilter: "+Person|-Admin",  // +包含, -排除
    minLevel: 1,
    maxLevel: 4,
    uniqueness: "NODE_GLOBAL"       // 不重复访问节点
  }
) YIELD path
RETURN path
```

---

## 六、子查询与组合查询

### 6.1 UNION

```cypher
// UNION 合并两个查询结果（列名必须一致）
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN p.name AS name, c.name AS entity, "employee" AS type
UNION
MATCH (p:Person)-[:STUDIES_AT]->(u:University)
RETURN p.name AS name, u.name AS entity, "student" AS type

// UNION ALL — 不去重（性能更好）
MATCH (p:Person)-[:BOUGHT]->(prod:Product)
RETURN p.name, prod.name AS item, "purchase" AS action
UNION ALL
MATCH (p:Person)-[:VIEWED]->(prod:Product)
RETURN p.name, prod.name AS item, "view" AS action
```

### 6.2 CALL IN TRANSACTIONS（大数据批量操作）

```cypher
// 对大量节点逐批处理（避免内存溢出）
MATCH (p:Person)
WHERE p.email IS NULL
CALL {
  WITH p
  SET p.email = toLower(p.name) + "@example.com",
      p.updated = datetime()
  RETURN count(*) AS updated
} IN TRANSACTIONS OF 1000 ROWS
RETURN sum(updated) AS total_updated

// 大数据 reindex
MATCH (p:OldPerson)
CALL {
  WITH p
  CREATE (new:Person {name: p.name, age: p.age})
  RETURN count(*) AS created
} IN TRANSACTIONS OF 5000 ROWS
RETURN sum(created)
```

---

## 七、Python neo4j 驱动实战

### 7.1 连接与基本查询

```python
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query, params=None):
        """执行只读查询"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query, params=None):
        """执行写入操作"""
        with self.driver.session() as session:
            result = session.execute_write(
                lambda tx: tx.run(query, params or {}).data()
            )
            return result

# 使用
client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")

# 参数化查询（防止注入）
result = client.execute_query(
    "MATCH (p:Person {name: $name})-[:WORKS_AT]->(c:Company) "
    "RETURN p.name, c.name, p.age ORDER BY p.age",
    {"name": "张三"}
)

# 批量创建
employees = [
    {"name": "张三", "age": 28, "company": "腾讯"},
    {"name": "李四", "age": 30, "company": "阿里巴巴"},
    {"name": "王五", "age": 25, "company": "腾讯"},
]

client.execute_write(
    "UNWIND $employees AS emp "
    "MERGE (c:Company {name: emp.company}) "
    "MERGE (p:Person {name: emp.name}) "
    "SET p.age = emp.age "
    "MERGE (p)-[:WORKS_AT]->(c)",
    {"employees": employees}
)
```

### 7.2 社交网络查询示例

```python
class SocialNetwork:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def add_friend(self, user1, user2, since=None):
        """添加好友关系"""
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    "MERGE (a:Person {name: $user1}) "
                    "MERGE (b:Person {name: $user2}) "
                    "MERGE (a)-[r:FRIENDS_WITH]->(b) "
                    "SET r.since = coalesce(r.since, date($since))",
                    user1=user1, user2=user2, 
                    since=since or "2024-01-01"
                )
            )
    
    def get_friends_of_friends(self, user, limit=10):
        """获取二度好友推荐"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (me:Person {name: $user})-[:FRIENDS_WITH*2]-(fof) "
                "WHERE fof <> me AND NOT (me)-[:FRIENDS_WITH]-(fof) "
                "RETURN fof.name AS name, count(*) AS mutual_count "
                "ORDER BY mutual_count DESC LIMIT $limit",
                user=user, limit=limit
            )
            return [dict(r) for r in result]
    
    def get_shortest_path(self, user1, user2):
        """查找两人之间的最短路径"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (a:Person {name: $user1}), (b:Person {name: $user2}) "
                "MATCH path = shortestPath((a)-[:FRIENDS_WITH*]-(b)) "
                "RETURN [n IN nodes(path) | n.name] AS path, "
                "       length(path) AS hops",
                user1=user1, user2=user2
            )
            record = result.single()
            return dict(record) if record else None
    
    def recommend_by_skills(self, user, top_k=5):
        """基于共同技能推荐好友"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (me:Person {name: $user})-[:HAS_SKILL]->(s:Skill) "
                "MATCH (other:Person)-[:HAS_SKILL]->(s) "
                "WHERE other <> me AND NOT (me)-[:FRIENDS_WITH]-(other) "
                "WITH other, count(DISTINCT s) AS common_skills, "
                "     collect(s.name) AS shared "
                "RETURN other.name AS name, common_skills, shared "
                "ORDER BY common_skills DESC LIMIT $k",
                user=user, k=top_k
            )
            return [dict(r) for r in result]

# 使用
sn = SocialNetwork("bolt://localhost:7687", "neo4j", "password")
fof = sn.get_friends_of_friends("张三")
path = sn.get_shortest_path("张三", "王五")
recs = sn.recommend_by_skills("张三")
```

### 7.3 事务与错误处理

```python
from neo4j import GraphDatabase
from neo4j.exceptions import TransientError
import time

def retry_on_transient(func, max_retries=3, delay=1):
    """处理 Neo4j 死锁等瞬态错误的重试装饰器"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except TransientError as e:
                if attempt == max_retries - 1:
                    raise
                print(f"瞬态错误: {e}, 重试 {attempt + 1}/{max_retries}")
                time.sleep(delay * (attempt + 1))
    return wrapper

# 使用显式事务控制
def transfer_friends(tx, from_user, to_user):
    """原子转移好友关系"""
    tx.run(
        "MATCH (from:Person {name: $from_user})-[r:FRIENDS_WITH]->(mutual) "
        "MATCH (to:Person {name: $to_user}) "
        "WHERE NOT (to)-[:FRIENDS_WITH]-(mutual) "
        "CREATE (to)-[:FRIENDS_WITH {since: date(), transferred: true}]->(mutual)",
        from_user=from_user, to_user=to_user
    )

with driver.session() as session:
    session.execute_write(transfer_friends, "张三", "李四")
```

---

## 八、性能分析与对比

### 8.1 Cypher vs SQL 查询对比

| 场景 | SQL (关系型) | Cypher (图数据库) | 优势方 |
|------|-------------|------------------|--------|
| 二度好友 | 多表 JOIN (2-3次) | `*2` 可变长度路径 | Cypher |
| 最短路径 | 需要应用层递归 | `shortestPath()` | Cypher |
| 按属性查询 | `WHERE id = X` (B-Tree) | 索引查找 | 平手 |
| 聚合统计 | GROUP BY | RETURN + 聚合函数 | SQL (更成熟) |
| 5+ 跳遍历 | JOIN 爆炸 | O(b^d) 遍历 | 都差，需限制深度 |

### 8.2 Cypher 查询性能参考

在 100 万节点、500 万关系的社交图上：

| 查询类型 | 延迟 | 说明 |
|---------|------|------|
| 单节点属性查询 (有索引) | ~1ms | B-Tree 查找 |
| 一度关系遍历 | ~2-5ms | 邻接表遍历 |
| 二度关系遍历 (avg_degree=50) | ~50-200ms | 2,500 次关系访问 |
| 三度关系遍历 (avg_degree=50) | ~2-10s | 125,000 次关系访问 |
| 全图 PageRank (GDS) | ~30-60s | 需要加载到内存 |

### 8.3 常见坑点

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 查询极慢 | 没有标签，触发全图扫描 | 始终指定标签 `MATCH (p:Person)` |
| MATCH 返回笛卡尔积 | 两个独立的 MATCH 没有路径连接 | 用 `WITH` 分阶段或合并为一个模式 |
| MERGE 并发死锁 | 多线程同时 MERGE 同一批节点 | 对输入排序，或用 `apoc.lock` |
| 可变长度路径爆炸 | 5+ 跳无限制遍历 | 限制最大深度，加 WHERE 过滤 |
| WHERE 写在 MATCH 里很慢 | 过滤条件被延迟到 Expand 之后 | 把属性条件放到 MATCH 的花括号里 |
