# SQL 临时表与公用表表达式


## 📋 SQL 临时表与公用表表达式


CREATE TEMPORARY TABLE 临时表、会话级 vs 事务级临时表、CTE WITH 子句、CTE vs 临时表选择、派生表子查询、临时表性能对比。


## 临时表基础


```
// ========== 临时表概念 ==========
-- 临时表: 只在当前会话/事务中存在的表
-- 自动删除 (会话结束或事务提交)

-- 用途:
-- 1. 存储中间结果
-- 2. 复杂查询分步处理
-- 3. 缓存重复使用的数据

-- ========== 创建临时表 ==========

-- 1. MySQL 临时表 (会话级)
CREATE TEMPORARY TABLE temp_high_value_users AS
SELECT id, name, email, total_spent
FROM users
WHERE total_spent > 1000;

-- 2. 创建带结构的临时表
CREATE TEMPORARY TABLE temp_order_summary (
    user_id INT,
    order_count INT,
    total_amount DECIMAL(12,2),
    last_order_date DATETIME
);

-- 插入数据
INSERT INTO temp_order_summary
SELECT user_id, COUNT(*), SUM(amount), MAX(created_at)
FROM orders
GROUP BY user_id;

-- 使用临时表
SELECT * FROM temp_order_summary
WHERE order_count > 5
ORDER BY total_amount DESC;

-- 3. PostgreSQL 临时表 (事务级默认)
-- CREATE TEMPORARY TABLE temp_data AS
-- SELECT * FROM source_table WHERE ...;

-- PostgreSQL 事务级临时表 (ON COMMIT DROP):
-- CREATE TEMPORARY TABLE temp_data (id INT) ON COMMIT DROP;

-- ========== 临时表 vs 普通表 ==========
-- 临时表:
--   - 自动删除 (会话/事务结束)
--   - 仅当前会话可见
--   - 不会创建 .frm 文件 (MySQL)
--   - 不同会话的同名临时表不冲突

-- 普通表:
--   - 持久存在
--   - 所有会话可见
--   - 需要手动 DROP
```


## 临时表实战


```
// ========== 实战 1: 多步骤数据处理 ==========

-- 分三步: 筛选 → 聚合 → 排名

-- 步骤 1: 找出高价值用户
CREATE TEMPORARY TABLE step1 AS
SELECT user_id, SUM(amount) AS total_spent
FROM orders
WHERE created_at >= '2024-01-01'
GROUP BY user_id
HAVING total_spent > 500;

-- 步骤 2: 获取用户详细信息
CREATE TEMPORARY TABLE step2 AS
SELECT u.id, u.name, u.email, s.total_spent
FROM step1 s
JOIN users u ON s.user_id = u.id
WHERE u.is_active = TRUE;

-- 步骤 3: 最终查询
SELECT *,
    ROW_NUMBER() OVER (ORDER BY total_spent DESC) AS rank
FROM step2
ORDER BY rank
LIMIT 10;

-- 临时表在会话结束时自动清理

-- ========== 实战 2: 避免重复 JOIN ==========

-- 不用临时表 (同一个 JOIN 多次):
SELECT * FROM products
WHERE category_id IN (
    SELECT id FROM categories WHERE is_active = TRUE
);
-- 同一子查询,可能执行多次

-- 用临时表 (一次 JOIN,多次使用):
CREATE TEMPORARY TABLE active_categories AS
SELECT id, name FROM categories WHERE is_active = TRUE;

SELECT p.* FROM products p
JOIN active_categories ac ON p.category_id = ac.id;

-- 后续还可以多次使用 active_categories

-- ========== 实战 3: 循环处理 ==========
-- 临时表 + 存储过程
-- 适合: 需要逐行处理的复杂业务逻辑

-- CREATE TEMPORARY TABLE ids_to_process AS
-- SELECT id FROM orders WHERE status = 'pending';
--
-- WHILE (EXISTS (SELECT 1 FROM ids_to_process)) DO
--     SELECT id INTO @current_id FROM ids_to_process LIMIT 1;
--     -- 处理逻辑...
--     DELETE FROM ids_to_process WHERE id = @current_id;
-- END WHILE;
```


## CTE vs 临时表


```
// ========== CTE (WITH 子句) ==========

-- CTE 优势:
-- 1. 不创建物理表 (只在内存中)
-- 2. 查询结束自动释放
-- 3. 可递归
-- 4. 可读性好

-- CTE 限制:
-- 1. 仅当前查询可用
-- 2. 某些数据库不物化 (多次引用会多次执行)
-- 3. 大数据量时性能不如临时表

-- ========== 何时用 CTE ==========

-- ✅ CTE 适用场景:
-- 1. 简单嵌套查询 → 提高可读性
WITH dept_avg AS (
    SELECT department, AVG(salary) AS avg_sal FROM employees GROUP BY department
)
SELECT * FROM employees e JOIN dept_avg d ON e.department = d.department;

-- 2. 递归查询 (树形结构)
WITH RECURSIVE org AS (...) ...

-- 3. 需要多次引用同一子查询
WITH stats AS (...)
SELECT * FROM stats WHERE ...;
SELECT * FROM stats WHERE ...;  -- 同一个查询,两次引用

-- ========== 何时用临时表 ==========

-- ✅ 临时表适用场景:
-- 1. 大数据量中间结果 (CTE 可能内存溢出)
-- 2. 多个查询之间共享数据
-- 3. 分步处理/调试
-- 4. 需要索引的中间结果

-- ========== 选择指南 ==========
-- 小数据 + 单查询     → CTE
-- 大数据 + 单查询     → CTE (如不会重复引用)
-- 多查询间共享数据    → 临时表
-- 树形查询            → 递归 CTE (唯一选择)
-- 分步处理            → 临时表
-- 需要索引的中间结果   → 临时表
-- 提高可读性          → CTE

-- ========== 派生表 (FROM 子查询) ==========
-- 另一种方式,但可读性差

-- ❌ 嵌套深:
SELECT * FROM (
    SELECT * FROM (
        SELECT * FROM employees WHERE salary > 50000
    ) AS high_salary
    WHERE department = 'Engineering'
) AS filtered;

-- ✅ CTE 更清晰:
WITH high_salary AS (
    SELECT * FROM employees WHERE salary > 50000
),
engineering AS (
    SELECT * FROM high_salary WHERE department = 'Engineering'
)
SELECT * FROM engineering;
```


## 性能对比与注意事项


```
// ========== 性能对比 ==========

-- CTE 物化: MySQL 8.0 可能物化 CTE
-- 使用 `EXPLAIN FORMAT=JSON` 查看

-- 如果 CTE 被多次引用:
WITH big_data AS (
    SELECT * FROM orders JOIN order_items ...
)
SELECT * FROM big_data WHERE...;
SELECT COUNT(*) FROM big_data;

-- MySQL 可能执行 big_data 两次!
-- 临时表只执行一次,然后重复使用

-- ========== 注意事项 ==========

-- 1. 临时表命名冲突
-- 避免与已有的普通表重名
-- 不同会话的同名临时表互不干扰

-- 2. 临时表索引
-- 大数据量临时表应建索引
CREATE TEMPORARY TABLE temp_big_data AS SELECT ...;
CREATE INDEX idx_temp_key ON temp_big_data(key_col);

-- 3. 临时表空间
-- MySQL: 临时表在 tmpdir 中
-- 监控: SHOW STATUS LIKE 'Created_tmp%';
-- 如果 Created_tmp_disk_tables 很大 → 临时表都在磁盘!

-- 4. 内存临时表
-- MySQL:
-- SET tmp_table_size = 64 * 1024 * 1024;  -- 64MB
-- SET max_heap_table_size = 64 * 1024 * 1024;
-- 超过大小 → 自动转为磁盘临时表 (变慢)

-- 5. PostgreSQL
-- 临时表在 temp_tablespaces 中
-- 可指定表空间: CREATE TEMP TABLE t() TABLESPACE temp2;
```


> **Note:** 💡 临时表要点: 会话结束自动删除; 适合大数据量中间结果; CTE 适合小数据 + 单查询; 多次引用的 CTE 可能多次执行; 临时表可建索引; 监控 Created_tmp_disk_tables 确保临时表在内存。


## 练习


<!-- Converted from: 27_SQL 临时表与公用表表达式.html -->
