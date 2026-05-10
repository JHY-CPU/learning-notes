# SQL 高级聚合与 PIVOT


## 📊 SQL 高级聚合与 PIVOT


条件聚合 (CASE WHEN + 聚合)、行转列 PIVOT、列转行 UNPIVOT、GROUPING SETS/CUBE/ROLLUP、分组内百分比、聚合过滤 FILTER (PostgreSQL)。


## 条件聚合 (行转列)


```
// ========== 条件聚合概念 ==========
-- 行转列: 将行中的值变成列名
-- 核心: CASE WHEN + SUM/COUNT

-- 原始数据 (行):
-- 日期       | 产品   | 销售额
-- 2024-01-01 | A     | 100
-- 2024-01-01 | B     | 150
-- 2024-01-02 | A     | 200
-- 2024-01-02 | B     | 120

-- 目标 (列):
-- 日期       | 产品A_销售额 | 产品B_销售额
-- 2024-01-01 | 100         | 150
-- 2024-01-02 | 200         | 120

-- ========== 1. 产品销售额透视 ==========
SELECT
    sale_date,
    SUM(CASE WHEN product = 'A' THEN amount ELSE 0 END) AS product_a,
    SUM(CASE WHEN product = 'B' THEN amount ELSE 0 END) AS product_b,
    SUM(CASE WHEN product = 'C' THEN amount ELSE 0 END) AS product_c,
    SUM(amount) AS total
FROM sales
GROUP BY sale_date
ORDER BY sale_date;

-- ========== 2. 各状态订单统计 ==========
SELECT
    user_id,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN status = 'pending'   THEN 1 ELSE 0 END) AS pending,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
    SUM(CASE WHEN status = 'refunded'  THEN 1 ELSE 0 END) AS refunded,
    ROUND(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS completion_pct
FROM orders
GROUP BY user_id;

-- ========== 3. 按月销售额 ==========
SELECT
    YEAR(created_at) AS year,
    SUM(CASE WHEN MONTH(created_at) = 1  THEN amount ELSE 0 END) AS Jan,
    SUM(CASE WHEN MONTH(created_at) = 2  THEN amount ELSE 0 END) AS Feb,
    SUM(CASE WHEN MONTH(created_at) = 3  THEN amount ELSE 0 END) AS Mar,
    SUM(CASE WHEN MONTH(created_at) = 4  THEN amount ELSE 0 END) AS Apr,
    SUM(CASE WHEN MONTH(created_at) = 5  THEN amount ELSE 0 END) AS May,
    SUM(CASE WHEN MONTH(created_at) = 6  THEN amount ELSE 0 END) AS Jun,
    SUM(CASE WHEN MONTH(created_at) = 7  THEN amount ELSE 0 END) AS Jul,
    SUM(CASE WHEN MONTH(created_at) = 8  THEN amount ELSE 0 END) AS Aug,
    SUM(CASE WHEN MONTH(created_at) = 9  THEN amount ELSE 0 END) AS Sep,
    SUM(CASE WHEN MONTH(created_at) = 10 THEN amount ELSE 0 END) AS Oct,
    SUM(CASE WHEN MONTH(created_at) = 11 THEN amount ELSE 0 END) AS Nov,
    SUM(CASE WHEN MONTH(created_at) = 12 THEN amount ELSE 0 END) AS Dec
FROM orders
GROUP BY year
ORDER BY year;
```


## PostgreSQL PIVOT / UNPIVOT


```
// ========== PostgreSQL crosstab ==========
-- PostgreSQL 通过 tablefunc 扩展实现 PIVOT

-- 1. 安装扩展
-- CREATE EXTENSION IF NOT EXISTS tablefunc;

-- 2. PIVOT (行转列)
-- SELECT * FROM crosstab(
--     'SELECT row_id, category, value FROM source_table ORDER BY 1,2',
--     'SELECT DISTINCT category FROM source_table ORDER BY 1'
-- ) AS ct (row_id INT, cat_a INT, cat_b INT, cat_c INT);

-- 示例: 产品月销售透视
-- SELECT * FROM crosstab(
--     'SELECT product, month, sales FROM product_sales ORDER BY 1,2',
--     'SELECT DISTINCT month FROM product_sales ORDER BY 1'
-- ) AS ct (product TEXT, jan NUMERIC, feb NUMERIC, mar NUMERIC);

-- ========== PostgreSQL FILTER ==========
-- PostgreSQL 的 FILTER 子句 (更简洁的条件聚合)

-- 传统条件聚合:
SELECT
    department,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE salary > 100000) AS high_salary,
    COUNT(*) FILTER (WHERE salary BETWEEN 50000 AND 100000) AS mid_salary,
    COUNT(*) FILTER (WHERE salary < 50000) AS low_salary,
    AVG(salary) FILTER (WHERE salary > 0) AS avg_salary
FROM employees
GROUP BY department;

-- 等价于 CASE WHEN,但语法更简洁!

-- ========== UNPIVOT (列转行) ==========
-- 将多列转成多行

-- 原始 (列):
-- id | q1 | q2 | q3 | q4
-- 1  | 100| 200| 150| 300

-- 目标 (行):
-- id | quarter | sales
-- 1  | Q1      | 100
-- 1  | Q2      | 200
-- 1  | Q3      | 150
-- 1  | Q4      | 300

-- MySQL UNPIVOT (使用 UNION ALL):
SELECT id, 'Q1' AS quarter, q1 AS sales FROM quarterly_sales
UNION ALL
SELECT id, 'Q2', q2 FROM quarterly_sales
UNION ALL
SELECT id, 'Q3', q3 FROM quarterly_sales
UNION ALL
SELECT id, 'Q4', q4 FROM quarterly_sales
ORDER BY id, quarter;

-- PostgreSQL UNPIVOT (更简洁):
-- SELECT id, quarter, sales
-- FROM quarterly_sales,
-- LATERAL (VALUES ('Q1', q1), ('Q2', q2), ('Q3', q3), ('Q4', q4)) AS v(quarter, sales);
```


## GROUPING SETS / CUBE / ROLLUP


```
// ========== GROUPING SETS ==========
-- 同时按多种维度分组,一次查询返回多个级别

-- 各部门各城市的薪资统计
SELECT
    department,
    city,
    SUM(salary) AS total_salary,
    COUNT(*) AS emp_count
FROM employees
GROUP BY GROUPING SETS (
    (department, city),  -- 部门+城市 级别
    (department),         -- 部门 小计
    (city),              -- 城市 小计
    ()                    -- 总计
);

-- ========== ROLLUP (层级汇总) ==========
-- 从最细到最粗的逐级汇总
-- ROLLUP(department, city) → (dept,city) + (dept) + 总计

SELECT
    department,
    city,
    SUM(salary) AS total_salary,
    GROUPING(department) AS is_dept_total,
    GROUPING(city) AS is_city_total
FROM employees
GROUP BY ROLLUP(department, city);

-- GROUPING() 函数: 1 表示该列是汇总行,0 是明细

-- 使用 COALESCE 美化:
SELECT
    COALESCE(department, '所有部门') AS department,
    COALESCE(city, '所有城市') AS city,
    SUM(salary) AS total
FROM employees
GROUP BY ROLLUP(department, city);

-- ========== CUBE (所有组合) ==========
-- 所有维度组合的汇总
-- CUBE(a, b, c) → 8 种组合

SELECT
    department,
    city,
    SUM(salary) AS total
FROM employees
GROUP BY CUBE(department, city);

-- 对比:
-- GROUP BY a, b              → 只有 (a,b)
-- ROLLUP(a, b)               → (a,b) + (a) + ()
-- CUBE(a, b)                 → (a,b) + (a) + (b) + ()
-- GROUPING SETS(a, b, ())    → (a) + (b) + ()
```


## 分组内百分比与占比


```
// ========== 分组内百分比 ==========

-- 1. 各部门人数占比
SELECT
    department,
    COUNT(*) AS emp_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM employees
GROUP BY department
ORDER BY pct DESC;

-- 2. 各部门薪资占比
SELECT
    department,
    SUM(salary) AS total_salary,
    ROUND(SUM(salary) * 100.0 / SUM(SUM(salary)) OVER (), 1) AS pct
FROM employees
GROUP BY department;

-- 3. 各组内占比 (PARTITION BY)
SELECT
    department,
    name,
    salary,
    ROUND(salary * 100.0 / SUM(salary) OVER (PARTITION BY department), 1) AS pct_in_dept
FROM employees;

-- ========== 累计百分比 ==========
-- 按销售额从高到低,计算每个产品的累计占比
WITH product_sales AS (
    SELECT
        product_id,
        SUM(amount) AS total_sales
    FROM order_items
    GROUP BY product_id
),
ranked AS (
    SELECT
        product_id,
        total_sales,
        SUM(total_sales) OVER (ORDER BY total_sales DESC) AS running_total,
        SUM(total_sales) OVER () AS grand_total
    FROM product_sales
)
SELECT
    product_id,
    total_sales,
    ROUND(total_sales * 100.0 / grand_total, 2) AS pct,
    ROUND(running_total * 100.0 / grand_total, 2) AS running_pct
FROM ranked
ORDER BY total_sales DESC;

-- ========== 二分位/四分位标记 ==========
-- NTILE 分组
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS quartile,
    CASE NTILE(4) OVER (ORDER BY salary)
        WHEN 1 THEN '低'
        WHEN 2 THEN '中低'
        WHEN 3 THEN '中高'
        WHEN 4 THEN '高'
    END AS level
FROM employees;
```


## 综合案例: 销售仪表板


```
// ========== 销售数据透视表 ==========

-- 每个产品类别 × 每个月的销售额
-- 行: 类别, 列: 月份
SELECT
    c.name AS category,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 1  THEN oi.quantity * oi.price END), 0) AS Jan,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 2  THEN oi.quantity * oi.price END), 0) AS Feb,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 3  THEN oi.quantity * oi.price END), 0) AS Mar,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 4  THEN oi.quantity * oi.price END), 0) AS Apr,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 5  THEN oi.quantity * oi.price END), 0) AS May,
    COALESCE(SUM(CASE WHEN MONTH(o.created_at) = 6  THEN oi.quantity * oi.price END), 0) AS Jun,
    SUM(oi.quantity * oi.price) AS total
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE YEAR(o.created_at) = 2024
GROUP BY c.id, c.name
ORDER BY total DESC;

-- ========== 多级汇总 ==========
SELECT
    COALESCE(c.name, '全部') AS category,
    COALESCE(p.name, '全部') AS product,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.price) AS revenue
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
WHERE oi.created_at >= '2024-01-01'
GROUP BY c.name, p.name WITH ROLLUP;

-- 结果:
-- 电子产品 | iPhone | 500
-- 电子产品 | MacBook| 300
-- 电子产品 | 全部   | 800       ← 小计
-- 服装     | T恤    | 1000
-- 服装     | 全部   | 1000      ← 小计
-- 全部     | 全部   | 1800      ← 总计
```


> **Note:** 💡 高级聚合: CASE WHEN + SUM 实现行转列; PostgreSQL FILTER 更简洁; GROUPING SETS 多维度汇总; ROLLUP 层级小计; CUBE 全组合; NTILE 分桶; 窗口函数计算分组内百分比。


## 练习


<!-- Converted from: 22_SQL 高级聚合与PIVOT.html -->
