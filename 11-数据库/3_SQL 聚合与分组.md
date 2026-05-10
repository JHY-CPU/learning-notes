# SQL 聚合与分组


## 📊 SQL 聚合与分组


聚合函数 (COUNT/SUM/AVG/MIN/MAX)、GROUP BY 分组、HAVING 过滤分组、GROUP_CONCAT/STRING_AGG、ROLLUP/CUBE。


## 聚合函数


```
// ========== 聚合函数 ==========
-- 聚合函数: 对多行数据计算返回单个值

-- COUNT: 计数
SELECT COUNT(*) FROM users;                 -- 总行数
SELECT COUNT(age) FROM users;               -- age 非 NULL 的行数
SELECT COUNT(DISTINCT city) FROM users;      -- 不同城市的数量

-- SUM: 求和
SELECT SUM(amount) FROM orders;
SELECT SUM(quantity * price) AS total FROM order_items;

-- AVG: 平均值
SELECT AVG(price) FROM products;
SELECT AVG(COALESCE(age, 0)) FROM users;    -- 处理 NULL

-- MIN / MAX: 最小值 / 最大值
SELECT MIN(price), MAX(price) FROM products;
SELECT MIN(created_at) AS first_order FROM orders;
SELECT MAX(salary) - MIN(salary) AS salary_range FROM employees;

-- ========== 综合统计 ==========
SELECT
    COUNT(*) AS total_orders,
    SUM(amount) AS revenue,
    AVG(amount) AS avg_order,
    MIN(amount) AS min_order,
    MAX(amount) AS max_order,
    COUNT(DISTINCT user_id) AS unique_customers
FROM orders
WHERE created_at >= '2024-01-01';

-- ========== 注意事项 ==========
-- 1. 聚合函数忽略 NULL (COUNT(*) 除外)
-- 2. AVG = SUM / COUNT (排除 NULL)
-- 3. COUNT(*) 包含 NULL 行, COUNT(col) 不包含
```


## GROUP BY 分组


```
// ========== GROUP BY ==========
-- GROUP BY: 将数据分组,每组应用聚合函数

-- 每个用户的订单数
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id;

-- 每个分类的商品数和平均价格
SELECT category_id,
       COUNT(*) AS product_count,
       AVG(price) AS avg_price
FROM products
GROUP BY category_id;

-- 按月统计销售额
SELECT
    DATE_FORMAT(created_at, '%Y-%m') AS month,
    SUM(amount) AS revenue
FROM orders
GROUP BY month
ORDER BY month;

-- 每个城市 + 状态的用户数 (多列分组)
SELECT city, status, COUNT(*) AS cnt
FROM users
GROUP BY city, status;

-- ========== GROUP BY 规则 ==========
-- 1. SELECT 中的非聚合列必须在 GROUP BY 中
-- ✅ 正确:
SELECT department, AVG(salary) FROM employees GROUP BY department;

-- ❌ 错误 (name 不在 GROUP BY 中):
-- SELECT department, name, AVG(salary) FROM employees GROUP BY department;

-- 2. 可以在 GROUP BY 中使用别名 (某些数据库)
SELECT DATE(created_at) AS day, COUNT(*) FROM orders GROUP BY day;

-- 3. 多个列组合分组
SELECT city, status, COUNT(*) FROM users GROUP BY 1, 2;
```


## HAVING 过滤分组


```
// ========== HAVING ==========
-- HAVING: 过滤分组结果 (类似 WHERE,但用于聚合后)
-- WHERE 在分组前过滤, HAVING 在分组后过滤

-- 找出订单数超过 5 的用户
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id
HAVING order_count > 5;

-- 找出平均消费 > 100 的用户
SELECT user_id, AVG(amount) AS avg_spent
FROM orders
GROUP BY user_id
HAVING avg_spent > 100;

-- WHERE + HAVING 组合
SELECT category_id,
       COUNT(*) AS cnt,
       AVG(price) AS avg_price
FROM products
WHERE price > 10           -- 分组前过滤低价商品
GROUP BY category_id
HAVING cnt >= 5             -- 分组后过滤商品少的分类
ORDER BY avg_price DESC;

-- ========== WHERE vs HAVING ==========
-- WHERE 在 GROUP BY 之前执行
-- HAVING 在 GROUP BY 之后执行
-- WHERE 不能使用聚合函数
-- HAVING 可以使用聚合函数

-- ❌ 错误:
-- SELECT department, AVG(salary) FROM employees
-- WHERE AVG(salary) > 50000 GROUP BY department;

-- ✅ 正确:
SELECT department, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING avg_salary > 50000;
```


## 字符串聚合


```
// ========== 字符串聚合 ==========
-- 将分组内的字符串连接成一个

-- MySQL: GROUP_CONCAT
SELECT department,
       GROUP_CONCAT(username ORDER BY username SEPARATOR ', ') AS members
FROM employees
GROUP BY department;

-- PostgreSQL: STRING_AGG
-- SELECT department, STRING_AGG(username, ', ' ORDER BY username) AS members
-- FROM employees GROUP BY department;

-- ========== ROLLUP / CUBE ==========
-- ROLLUP: 多级分组汇总 (小计+总计)

-- MySQL:
SELECT
    COALESCE(category, '全部') AS category,
    COALESCE(brand, '全部') AS brand,
    SUM(quantity) AS total_qty
FROM sales
GROUP BY category, brand WITH ROLLUP;

-- PostgreSQL:
-- GROUP BY ROLLUP (category, brand)

-- CUBE: 所有维度组合的汇总
-- GROUP BY CUBE (category, brand)

-- ========== 实用示例 ==========
-- 每个分类销量前 3 的商品
SELECT category_id, product_id, SUM(qty) AS total_sold
FROM order_items
GROUP BY category_id, product_id
ORDER BY category_id, total_sold DESC;

-- 每日活跃用户数 (过去 7 天)
SELECT DATE(login_time) AS day, COUNT(DISTINCT user_id) AS dau
FROM user_logins
WHERE login_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY day
ORDER BY day;
```


## 完整示例: 销售分析


```
// ========== 销售数据分析 ==========
-- 订单表: orders(id, user_id, amount, created_at, status)
-- 订单明细: order_items(id, order_id, product_id, quantity, price)
-- 商品表: products(id, name, category_id, price)
-- 用户表: users(id, name, city, created_at)

-- 1. 每月销售额趋势
SELECT
    DATE_FORMAT(o.created_at, '%Y-%m') AS month,
    COUNT(DISTINCT o.id) AS orders,
    SUM(o.amount) AS revenue,
    AVG(o.amount) AS avg_order_value
FROM orders o
WHERE o.status = 'completed'
GROUP BY month
ORDER BY month;

-- 2. 每个城市的消费总额和用户数
SELECT
    u.city,
    COUNT(DISTINCT u.id) AS user_count,
    COUNT(DISTINCT o.id) AS order_count,
    SUM(o.amount) AS total_revenue,
    ROUND(SUM(o.amount) / COUNT(DISTINCT u.id), 2) AS revenue_per_user
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'completed'
GROUP BY u.city
ORDER BY total_revenue DESC;

-- 3. 畅销商品 TOP 10
SELECT
    p.id,
    p.name,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.price) AS revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.status = 'completed'
GROUP BY p.id, p.name
ORDER BY units_sold DESC
LIMIT 10;

-- 4. 高价值用户 (累计消费 > 1000)
SELECT
    u.id,
    u.name,
    COUNT(o.id) AS order_count,
    SUM(o.amount) AS total_spent,
    MAX(o.created_at) AS last_order
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.status = 'completed'
GROUP BY u.id, u.name
HAVING total_spent > 1000
ORDER BY total_spent DESC;
```


> **Note:** 💡 聚合与分组: COUNT/SUM/AVG/MIN/MAX 聚合函数; GROUP BY 分组; HAVING 过滤分组; WHERE 先于 GROUP BY 执行; GROUP_CONCAT 字符串聚合; ROLLUP 多级汇总。


## 练习


<!-- Converted from: 3_SQL 聚合与分组.html -->
