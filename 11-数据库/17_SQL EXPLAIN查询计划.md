# SQL EXPLAIN 查询计划


## 🔬 SQL EXPLAIN 查询计划


EXPLAIN 输出解读、type 列访问类型、Extra 信息、rows 估算、key 使用索引、EXPLAIN ANALYZE (PostgreSQL)、慢查询优化实战案例。


## EXPLAIN 基础


```
// ========== EXPLAIN 概念 ==========
-- EXPLAIN: 显示 MySQL 如何执行查询
-- 不实际执行查询,只显示执行计划
-- 是 SQL 优化最重要的工具!

-- 使用方法:
EXPLAIN SELECT * FROM users WHERE email = 'test@test.com';
-- 或更详细:
EXPLAIN FORMAT=JSON SELECT * FROM users WHERE email = 'test@test.com';

-- ========== 输出列 ==========
-- id:           查询序号 (子查询递增)
-- select_type:  查询类型
-- table:        表名
-- partitions:   涉及分区
-- type:         访问类型 ← 最重要的列!
-- possible_keys: 可能使用的索引
-- key:           实际使用的索引
-- key_len:       索引使用长度
-- ref:           索引比较的列/常量
-- rows:          估计扫描行数
-- filtered:     过滤后百分比
-- Extra:        额外信息 ← 第二重要的列!

-- ========== 示例 ==========
EXPLAIN SELECT u.name, o.amount
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.email = 'alice@test.com';

-- 输出示例:
-- id | select_type | table | type  | key             | rows | Extra
-- 1  | SIMPLE      | u     | const | idx_users_email | 1    | Using index
-- 1  | SIMPLE      | o     | ref   | idx_orders_user | 3    | NULL

-- 解读:
-- users 表: 用唯一索引 (const), 扫描 1 行, 使用覆盖索引
-- orders 表: 用普通索引 (ref), 扫描 3 行
```


## type 列详解


```
// ========== type 访问类型 (好→差) ==========

-- system: 表只有一行 (系统表)
-- 最好,几乎不可能达到
EXPLAIN SELECT * FROM mysql.user LIMIT 1;

-- const: 主键/唯一索引等值查询
-- 最多返回一行,非常快
EXPLAIN SELECT * FROM users WHERE id = 1;

-- eq_ref: JOIN 中使用主键/唯一索引
-- 每行只匹配一行,JOIN 中最优
EXPLAIN SELECT * FROM users u
JOIN profiles p ON u.id = p.user_id;  -- p.user_id 是主键/唯一键

-- ref: 非唯一索引等值查询
-- 可能返回多行,但仍然高效
EXPLAIN SELECT * FROM orders WHERE user_id = 1;

-- range: 范围查询
-- > < BETWEEN IN LIKE 前缀
EXPLAIN SELECT * FROM users WHERE age BETWEEN 20 AND 30;
EXPLAIN SELECT * FROM users WHERE email LIKE 'alice%';

-- index: 扫描整个索引树
-- 比全表快一些 (索引比表小)
EXPLAIN SELECT id, email FROM users ORDER BY email;
-- Extra 可能显示 Using index

-- ALL: 全表扫描 ← 最差!
-- 需要优化! 加索引或改写查询
EXPLAIN SELECT * FROM users WHERE name = 'Alice';
-- 如果 name 没有索引 → ALL

-- ========== 优化目标 ==========
-- 理想: const / eq_ref / ref / range
-- 可接受: index
-- 需优化: ALL
```


## Extra 列详解


```
// ========== Extra 重要信息 ==========

-- 1. Using index (覆盖索引)
-- 索引包含所有需要的列,不用回表
-- 非常好!
EXPLAIN SELECT id, email FROM users WHERE email = 'test@test.com';
-- 如果索引在 (id, email) 上

-- 2. Using index condition (索引下推 ICP)
-- 在索引层过滤,减少回表次数
-- MySQL 5.6+ 优化,好

-- 3. Using where
-- 回表后过滤数据
-- 说明索引不够"覆盖"

-- 4. Using filesort (文件排序)
-- 需要额外排序操作
-- 可能需要优化索引或查询
EXPLAIN SELECT * FROM users ORDER BY created_at;
-- 如果 created_at 没有索引 → filesort

-- 5. Using temporary (临时表)
-- 使用临时表存储中间结果
-- 常见于 GROUP BY / DISTINCT
-- 性能差,需要优化!
EXPLAIN SELECT DISTINCT city FROM users GROUP BY city;
-- 如果 city 没有索引 → temporary

-- 6. Using join buffer (JOIN 缓存)
-- JOIN 时没有用索引
-- 需要为 JOIN 列加索引!
EXPLAIN SELECT * FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
-- 如果 o.user_id 没有索引 → Using join buffer

-- 7. Impossible WHERE (不可能的条件)
EXPLAIN SELECT * FROM users WHERE id = 1 AND id = 2;

-- ========== 需要警惕的 Extra ==========
-- ❌ Using filesort   → 加排序索引
-- ❌ Using temporary  → 优化 GROUP BY/DISTINCT
-- ❌ Using join buffer → JOIN 列加索引
```


## EXPLAIN 实战案例


```
// ========== 案例 1: 慢查询优化 ==========

-- 问题: 查询速度很慢
SELECT * FROM orders
WHERE user_id = 1
ORDER BY created_at DESC
LIMIT 10;

-- EXPLAIN 结果:
-- type: ALL, rows: 100000, Extra: Using filesort
-- ❌ 全表扫描 + 文件排序!

-- 优化: 加复合索引
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- 再次 EXPLAIN:
-- type: ref, rows: 50, Extra: Using index condition
-- ✅ 索引查找,无需额外排序!

-- ========== 案例 2: 覆盖索引 ==========

-- 查询所有用户的 email 和创建时间
SELECT email, created_at FROM users;

-- 如果有一个索引 (email, created_at):
-- Extra: Using index ✅ 完全不需要回表!

-- ========== 案例 3: JOIN 优化 ==========

-- 慢查询:
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- EXPLAIN:
-- orders 表: type: ALL, Extra: Using join buffer
-- ❌ 全表扫描 JOIN!

-- 优化: 在 o.user_id 上加索引
CREATE INDEX idx_orders_user ON orders(user_id);

-- 再次 EXPLAIN:
-- type: ref ✅

-- ========== 案例 4: 分页优化验证 ==========

-- 传统分页 (大偏移量慢)
EXPLAIN SELECT * FROM orders ORDER BY id LIMIT 100000, 20;
-- rows: 100020 (扫描所有行!)

-- 延迟关联分页
EXPLAIN SELECT o.* FROM orders o
JOIN (SELECT id FROM orders ORDER BY id LIMIT 100000, 20) tmp
  ON o.id = tmp.id;
-- 子查询: rows: 100020
-- 外层:   type: eq_ref (主键查找,极快)
```


## EXPLAIN ANALYZE (PostgreSQL)


```
// ========== PostgreSQL EXPLAIN ==========

-- 1. EXPLAIN (只显示计划,不执行)
EXPLAIN SELECT * FROM users WHERE email = 'test@test.com';

-- 2. EXPLAIN ANALYZE (实际执行!)
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@test.com';

-- 输出示例:
-- Seq Scan on users  (cost=0.00..12.10 rows=1 width=36)
--   (actual time=0.015..0.016 rows=1 loops=1)
--   Filter: (email = 'test@test.com'::text)
--   Rows Removed by Filter: 999
-- Planning Time: 0.05 ms
-- Execution Time: 0.02 ms

-- 解读 PostgreSQL 输出:
-- cost:      估算成本 (启动..总成本)
-- rows:      估算行数
-- actual:    实际执行时间、行数、循环次数
-- loops:    循环执行次数
-- Planning:  计划生成时间
-- Execution: 实际执行时间

-- PostgreSQL 访问方式:
-- Seq Scan:    顺序扫描 (全表) ← 差
-- Index Scan:  索引扫描
-- Index Only Scan: 覆盖索引
-- Bitmap Scan: 位图扫描 (多个索引组合)
-- Nested Loop: 嵌套循环 JOIN
-- Hash Join:   哈希 JOIN
-- Merge Join:  归并 JOIN

-- ========== MySQL EXPLAIN ANALYZE (8.0.18+) ==========
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@test.com';
-- 类似 PostgreSQL,提供实际执行时间和行数

-- ========== 格式选项 ==========
-- MySQL:
EXPLAIN FORMAT=TRADITIONAL SELECT ...;
EXPLAIN FORMAT=JSON SELECT ...;
EXPLAIN FORMAT=TREE SELECT ...;  -- 8.0.16+

-- SHOW WARNINGS 查看优化后的 SQL:
EXPLAIN SELECT ...;
SHOW WARNINGS;
```


## 优化方法论


```
// ========== SQL 优化步骤 ==========

// 1. 找到慢查询
//    MySQL: 开启慢查询日志
//    SET GLOBAL slow_query_log = ON;
//    SET GLOBAL long_query_time = 1;  -- >1秒

//    PostgreSQL: 配置 log_min_duration_statement

// 2. 分析查询计划
//    EXPLAIN 慢查询
//    关注: type, rows, Extra

// 3. 识别问题
//    - type = ALL → 缺少索引
//    - Using filesort → 排序优化
//    - Using temporary → 分组优化
//    - rows 很大 → 需要更好的过滤

// 4. 优化
//    - 添加合适的索引
//    - 改写查询
//    - 调整表结构

// 5. 验证效果
//    再次 EXPLAIN 对比
//    测试实际执行时间

// ========== 性能指标参考 ==========
// 行数     | 优化目标
// < 1000   | type = ref 以上
// 1000-1w  | type = ref, 无 filesort
// 1w-10w   | 覆盖索引
// > 10w    | 范围分区或分表

// ========== 常见误区 ==========
// 1. 只看 rows,不看 type
// 2. 索引越多越好 (错!)
// 3. EXPLAIN 结果不变就不慢 (错!)
// 4. 不关注 Extra
// 5. 忽略 filtered 列
```


> **Note:** 💡 EXPLAIN 要点: type=ALL 需要优化; Extra 注意 Using filesort/temporary/join buffer; key 为 NULL 表示没走索引; rows 估算扫描行数; 优化后对比 EXPLAIN 验证效果; 慢查询日志是发现问题的起点。


## 练习


<!-- Converted from: 17_SQL EXPLAIN查询计划.html -->
