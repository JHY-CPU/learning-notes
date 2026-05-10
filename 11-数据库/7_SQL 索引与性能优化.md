# SQL 索引与性能优化


## ⚡ SQL 索引与性能优化


B-Tree 索引结构、复合索引最左前缀、EXPLAIN 执行计划、覆盖索引、索引扫描 vs 全表扫描、索引下推、查询优化技巧。


## 索引基础


```
// ========== 索引概念 ==========
-- 索引: 加速数据检索的数据结构
-- 类似书的目录 → 不用翻遍全书
-- 原理: 以空间换时间

-- 索引类型:
-- 1. B-Tree 索引   → 最常用,<>= BETWEEN IN ORDER BY
-- 2. Hash 索引     → = 查询极快,不支持范围
-- 3. 全文索引      → LIKE 搜索替代,文本匹配
-- 4. 空间索引      → GIS 地理数据
-- 5. GiST/GIN      → PostgreSQL 高级索引

-- ========== B-Tree 结构 ==========
--          [50]
--         /    \
--     [20,30]  [70,90]
--    /   |   \   |   \
--   ...  ...  ... ... ...

-- 特点:
-- - 平衡树: 所有叶子节点同层
-- - 有序: 中序遍历得到排序结果
-- - O(log N) 查找复杂度
-- - 支持范围查询 (BETWEEN, >, <)

-- ========== 创建索引 ==========
-- 单列索引
CREATE INDEX idx_users_email ON users(email);

-- 唯一索引
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- 复合索引 (多列)
CREATE INDEX idx_users_city_age ON users(city, age);

-- 部分索引 (PostgreSQL)
-- CREATE INDEX idx_active_users ON users(email) WHERE is_active = TRUE;

-- 降序索引
CREATE INDEX idx_orders_date_desc ON orders(created_at DESC);

-- 删除索引
DROP INDEX idx_users_email ON users;          -- MySQL
-- DROP INDEX idx_users_email;                -- PostgreSQL
```


## 复合索引与最左前缀


```
// ========== 复合索引最左前缀 ==========
-- 复合索引: (city, age, name)

-- ✅ 走索引:
WHERE city = 'Beijing'                    -- 第一列
WHERE city = 'Beijing' AND age = 25       -- 第一列+第二列
WHERE city = 'Beijing' AND age = 25 AND name = 'Alice'  -- 全匹配
WHERE city IN ('Beijing', 'Shanghai')     -- 第一列 (IN)

-- ❌ 不走索引:
WHERE age = 25                            -- 跳过了第一列
WHERE name = 'Alice'                      -- 跳过了第一列
WHERE age = 25 AND name = 'Alice'         -- 跳过了第一列

-- ⚠️ 部分走索引:
WHERE city = 'Beijing' AND name = 'Alice' -- 只用 city 列,跳过 age
-- 索引只用 city 部分过滤, name 在索引查找后过滤

-- ========== 索引选择性 ==========
-- 选择性 = COUNT(DISTINCT col) / COUNT(*)
-- 选择性越高 → 索引越有效

-- 高选择性 (好):
-- email, 身份证号 → 接近 1.0

-- 低选择性 (差):
-- gender (男/女)  → ~0.5
-- is_active (0/1) → 很低

-- 经验: 选择性 > 0.1 才适合建索引

-- ========== 索引设计原则 ==========
-- 1. 经常出现在 WHERE 中的列
-- 2. 频繁 JOIN 的关联列 (外键)
-- 3. ORDER BY / GROUP BY 的列
-- 4. 选择性高的列优先
-- 5. 不要过度索引 (写操作变慢)
-- 6. 短索引优于长索引 (VARCHAR(255) 不如 VARCHAR(50))
```


## EXPLAIN 执行计划


```
// ========== EXPLAIN ==========
-- 查看 MySQL 执行计划
EXPLAIN SELECT * FROM users WHERE email = 'alice@test.com'\G

-- 输出关键字段:
-- id:        查询序号 (子查询 id 递增)
-- select_type: SIMPLE / PRIMARY / SUBQUERY / DERIVED
-- table:     表名
-- type:      访问类型 (性能从好到差)
--   system:  表只有一行 (系统表)
--   const:   主键/唯一索引等值查询
--   eq_ref:  JOIN 主键/唯一索引
--   ref:     非唯一索引等值查询
--   range:   范围查询 (>, <, BETWEEN, IN)
--   index:   扫描索引树 (全索引扫描)
--   ALL:     全表扫描 ← 性能差!
-- possible_keys: 可能用的索引
-- key:          实际用的索引
-- key_len:      索引使用的字节数
-- rows:         估计扫描行数
-- Extra:        额外信息
--   Using index:       覆盖索引 (无需回表)
--   Using where:       回表后过滤
--   Using index condition: 索引下推 (ICP)
--   Using filesort:    需要额外排序
--   Using temporary:   需要临时表 (GROUP BY/ DISTINCT)

-- ========== 执行计划解读 ==========
-- 示例:
EXPLAIN SELECT u.name, o.amount
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.email = 'alice@test.com'\G

-- 理想情况:
-- users 表: type=const (主键/唯一索引)
-- orders 表: type=ref (索引查找)

-- 糟糕情况:
-- users 表: type=ALL (全表扫描!)
-- orders 表: type=ALL

-- PostgreSQL: EXPLAIN ANALYZE (实际执行)
-- EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'alice@test.com';
```


## 覆盖索引与索引下推


```
// ========== 覆盖索引 ==========
-- 索引包含查询所需的所有列
-- 无需回表访问数据行 → 性能极大提升

-- 示例: 复合索引 (city, age, name)

-- 覆盖索引 (Extra: Using index):
SELECT city, age FROM users WHERE city = 'Beijing';
-- 索引中已有 city, age, 无需回表

-- 非覆盖索引 (Extra: Using index condition):
SELECT city, age, email FROM users WHERE city = 'Beijing';
-- email 不在索引中,需要回表

-- 设计覆盖索引:
-- 查询频繁的列尽量包含在索引中
-- 但不要把所有列都放进索引 (太大)

-- ========== 索引下推 (ICP) ==========
-- Index Condition Pushdown
-- MySQL 5.6+ 优化
-- 将 WHERE 条件过滤"下推"到索引层

-- 复合索引 (city, age)
SELECT * FROM users
WHERE city = 'Beijing'
  AND age > 20
  AND name LIKE '%san%';  -- name 不在索引中

-- 无 ICP: 找到 city='Beijing' AND age>20 的行,回表,再过滤 name
-- 有 ICP: 在索引层就过滤 name (Using index condition),减少回表

-- ========== 索引下推图示 ==========
-- 传统: 索引查找 → 回表 (完整行) → WHERE 过滤
-- ICP:  索引查找+部分过滤 → 回表 (减少回表次数)

-- ========== MRR (Multi-Range Read) ==========
-- 将随机 I/O 转为顺序 I/O
-- 先排序主键,再回表读取
-- 对辅助索引的范围扫描有效
```


## 查询优化实战


```
// ========== 常见优化技巧 ==========

-- 1. 避免 SELECT *
-- ❌:
SELECT * FROM users WHERE email = 'test@test.com';
-- ✅ (覆盖索引):
SELECT id, email FROM users WHERE email = 'test@test.com';
-- (前提: 有 (email, id) 复合索引)

-- 2. LIKE 优化
-- ❌ 无法使用索引:
SELECT * FROM users WHERE email LIKE '%alice%';
-- ✅ 可以使用索引 (前缀匹配):
SELECT * FROM users WHERE email LIKE 'alice%';

-- 3. 函数包裹列 → 无法使用索引
-- ❌:
SELECT * FROM orders WHERE DATE(created_at) = '2024-01-15';
-- ✅:
SELECT * FROM orders WHERE created_at >= '2024-01-15 00:00:00'
                        AND created_at < '2024-01-16 00:00:00';

-- 4. 隐式类型转换
-- ❌ (phone 是 VARCHAR,但传入数字):
SELECT * FROM users WHERE phone = 13800138000;
-- 隐式类型转换使索引失效
-- ✅:
SELECT * FROM users WHERE phone = '13800138000';

-- 5. OR 优化
-- ❌:
SELECT * FROM users WHERE city = 'Beijing' OR age = 25;
-- ✅ (用 UNION 替代 OR):
SELECT * FROM users WHERE city = 'Beijing'
UNION
SELECT * FROM users WHERE age = 25;

-- 6. 分页优化 (延迟关联)
-- ❌ 大偏移量:
SELECT * FROM orders ORDER BY id LIMIT 100000, 20;
-- ✅ 延迟关联:
SELECT o.*
FROM orders o
JOIN (SELECT id FROM orders ORDER BY id LIMIT 100000, 20) AS tmp
  ON o.id = tmp.id;

-- 7. 避免在 WHERE 中对列做运算
-- ❌:
SELECT * FROM orders WHERE amount * 1.1 > 100;
-- ✅:
SELECT * FROM orders WHERE amount > 100 / 1.1;
```


## 索引维护与监控


```
// ========== 索引维护 ==========

-- 1. 查看索引
-- MySQL:
SHOW INDEX FROM users;

-- PostgreSQL:
-- \d users

-- 2. 索引碎片整理
-- MySQL:
ALTER TABLE users ENGINE=InnoDB;      -- 重建表
OPTIMIZE TABLE users;                 -- 优化表

-- PostgreSQL:
-- REINDEX INDEX idx_users_email;
-- VACUUM ANALYZE users;

-- 3. 未使用索引检测
-- MySQL (慢查询日志):
-- SET GLOBAL slow_query_log = ON;
-- SET GLOBAL long_query_time = 2;  -- 超过 2 秒

-- 4. 索引使用统计
-- MySQL:
-- SELECT * FROM performance_schema.table_io_waits_summary_by_index_usage;

-- ========== 索引权衡 ==========
-- 优点: 大幅加速 SELECT
-- 缺点: 减慢 INSERT/UPDATE/DELETE
--       占用磁盘空间
--       维护成本

-- 何时不该建索引:
-- - 小表 (< 1000 行)
-- - 频繁大批量写入的表
-- - 选择性极低的列 (gender, is_active)
-- - 几乎不查询的列

-- ========== 经验法则 ==========
-- 1. 数据量 < 1000: 不需要索引 (全表扫描够快)
-- 2. 数据量 > 10000: 需要精心设计索引
-- 3. EXPLAIN 是优化最重要的工具
-- 4. 索引不是越多越好
-- 5. 监控慢查询日志
-- 6. 定期分析查询模式并调整索引
```


> **Note:** 💡 索引要点: B-Tree 支持范围查询; 复合索引遵循最左前缀; EXPLAIN 看 type/rows/Extra; 覆盖索引避免回表; 索引下推 ICP 减少回表; 避免函数运算、隐式转换导致索引失效; 大分页用延迟关联优化。


## 练习


<!-- Converted from: 7_SQL 索引与性能优化.html -->
