# SQL 高级查询技巧


## 🚀 SQL 高级查询技巧


SELECT 高级用法、FIRST/LAST 每组取首末、Pagination 游标分页、随机抽样 (TABLESAMPLE)、间隔查询、滑动聚合、高级条件过滤技巧。


## 每组 Top-N


```
// ========== 每组前 N ==========

-- 场景: 每个部门薪资最高的 3 人

-- MySQL 8.0+ / PostgreSQL:
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY department ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT name, department, salary
FROM ranked
WHERE rn <= 3;

-- MySQL 5.x (用户变量):
SELECT name, department, salary
FROM (
    SELECT *,
        @rn := IF(@dept = department, @rn + 1, 1) AS rn,
        @dept := department
    FROM employees
    CROSS JOIN (SELECT @dept := NULL, @rn := 0) vars
    ORDER BY department, salary DESC
) ranked
WHERE rn <= 3;

-- ========== 每组第一名/最后一名 ==========

-- 每个分类最新商品 (FIRST VALUE / LAST VALUE)
SELECT DISTINCT
    c.name AS category,
    FIRST_VALUE(p.name) OVER (
        PARTITION BY c.id ORDER BY p.created_at DESC
    ) AS latest_product,
    FIRST_VALUE(p.created_at) OVER (
        PARTITION BY c.id ORDER BY p.created_at DESC
    ) AS latest_date
FROM categories c
JOIN products p ON c.id = p.category_id;

-- 或用 LATERAL (更高效):
SELECT c.name AS category, top.name, top.created_at
FROM categories c
LEFT JOIN LATERAL (
    SELECT p.name, p.created_at
    FROM products p
    WHERE p.category_id = c.id
    ORDER BY p.created_at DESC
    LIMIT 1
) top ON true;
```


## 游标分页 (Keyset Pagination)


```
// ========== 游标分页 ==========
-- 传统 LIMIT/OFFSET 大偏移量越来越慢
-- 游标分页: 基于上一页最后一条记录的位置

-- 传统分页 (越往后越慢):
SELECT * FROM orders
ORDER BY id
LIMIT 20 OFFSET 100000;

-- 游标分页 (始终快):
-- 第 1 页:
SELECT * FROM orders
ORDER BY id
LIMIT 20;

-- 第 2 页 (上一页最后 id = 20):
SELECT * FROM orders
WHERE id > 20
ORDER BY id
LIMIT 20;

-- 第 N 页 (上一页最后 id = last_id):
SELECT * FROM orders
WHERE id > :last_id
ORDER BY id
LIMIT 20;

-- 多列游标 (ORDER BY created_at, id):
SELECT * FROM orders
WHERE (created_at, id) > (:last_date, :last_id)
ORDER BY created_at, id
LIMIT 20;

-- ========== 优缺点 ==========
-- 优点: 大偏移量一样快, O(1)
-- 缺点: 不支跳转到任意页 (只能上一页/下一页)
-- 适用: 无限滚动, "加载更多"
```


## 抽样与随机


```
// ========== 随机抽样 ==========

-- 1. 随机排序 (数据量大时很慢!)
SELECT * FROM users ORDER BY RAND() LIMIT 10;

-- 2. 更快的方式 (使用主键范围随机)
SELECT * FROM users
WHERE id >= (
    SELECT FLOOR(RAND() * (MAX(id) - MIN(id)) + MIN(id))
    FROM users
)
LIMIT 10;

-- 但可能少于 10 行 (ID 不连续)

-- 3. PostgreSQL TABLESAMPLE (快!)
-- SELECT * FROM users TABLESAMPLE BERNOULLI(1) LIMIT 10;
-- SELECT * FROM users TABLESAMPLE SYSTEM(1) LIMIT 10;

-- BERNOULLI: 逐行概率 (均匀)
-- SYSTEM: 按页抽样 (更快,但偏差)

-- 4. MySQL 8.0: 无内置抽样,用 RAND() 或外部处理

-- ========== 间隔查询 ==========

-- 每第 N 行
SET @row_num = 0;
SELECT * FROM (
    SELECT *, @row_num := @row_num + 1 AS rn
    FROM users
    ORDER BY id
) numbered
WHERE rn % 10 = 0;  -- 每 10 行取一行

-- 时间间隔聚合 (每 5 分钟)
SELECT
    FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(created_at) / 300) * 300) AS interval_start,
    COUNT(*) AS count
FROM events
GROUP BY interval_start
ORDER BY interval_start;
```


## 实用高级技巧


```
// ========== 技巧 1: 跳过行 + 取 N ==========

-- 每个分类跳过前 2 名,取第 3-5 名
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY department ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT name, department, salary, rn
FROM ranked
WHERE rn BETWEEN 3 AND 5;

-- ========== 技巧 2: 条件 HAVING ==========

-- 有订单的用户中, 总消费 > 1000 并且至少有 2 个完成的订单
SELECT user_id,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_spent,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_orders
FROM orders
GROUP BY user_id
HAVING total_spent > 1000
   AND completed_orders >= 2;

-- ========== 技巧 3: 排除逻辑 ==========

-- 购买了 A 但没有购买 B 的用户
SELECT DISTINCT o1.user_id
FROM orders o1
JOIN order_items oi1 ON o1.id = oi1.order_id
JOIN products p1 ON oi1.product_id = p1.id
WHERE p1.name = 'A'
  AND NOT EXISTS (
    SELECT 1 FROM orders o2
    JOIN order_items oi2 ON o2.id = oi2.order_id
    JOIN products p2 ON oi2.product_id = p2.id
    WHERE o2.user_id = o1.user_id AND p2.name = 'B'
  );

-- ========== 技巧 4: 加权平均 ==========

-- 计算加权平均分
SELECT
    SUM(score * weight) / SUM(weight) AS weighted_avg
FROM exam_scores;

-- ========== 技巧 5: Running Diff ==========
SELECT
    date,
    revenue,
    revenue - LAG(revenue) OVER (ORDER BY date) AS daily_change,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY date))
        / LAG(revenue) OVER (ORDER BY date) * 100, 2) AS daily_change_pct
FROM daily_revenue;
```


> **Note:** 💡 高级技巧: Top-N 用 ROW_NUMBER + WHERE rn <= N; 游标分页用 WHERE id > last_id; RAND() 随机但慢; TABLESAMPLE 快速抽样 (PG); 间隔查询按时间窗口分组; 条件 HAVING 支持复杂过滤; LAG/LEAD 做差值分析。


## 练习


<!-- Converted from: 34_SQL 高级查询技巧.html -->
