# SQL JOIN连接


## 🔗 SQL JOIN 连接


INNER/LEFT/RIGHT/FULL JOIN、自连接、多表 JOIN 执行顺序、笛卡尔积 CROSS JOIN、LATERAL JOIN。


## JOIN 类型概览


```
// ========== JOIN 类型 ==========
-- 数据:
-- users:                    orders:
-- id | name                 id | user_id | amount
-- 1  | Alice               1   | 1      | 100
-- 2  | Bob                 2   | 1      | 200
-- 3  | Carol               3   | 2      | 150
-- 4  | Dave                (Dave 无订单)

-- INNER JOIN: 交集 (两边都匹配的行)
-- Alice(1) → 订单1, 订单2
-- Bob(2)   → 订单3
-- Carol(3) → 无 (不包含)
-- Dave(4)  → 无 (不包含)

-- LEFT JOIN: 左表所有行 + 右表匹配行
-- Alice(1) → 订单1, 订单2
-- Bob(2)   → 订单3
-- Carol(3) → NULL
-- Dave(4)  → NULL

-- RIGHT JOIN: 右表所有行 + 左表匹配行 (很少用,LEFT 可替代)
-- FULL JOIN: 并集 (左右所有行) (MySQL 不支持,用 UNION 替代)

-- ========== 文氏图 ==========
--           INNER JOIN        LEFT JOIN         RIGHT JOIN       FULL JOIN
--           ┌─────┐          ┌─────┐           ┌─────┐          ┌─────┐
--           │  A  │          │░░░A░│           │  A  │          │░░░A░│
--    ┌───┐  │ ┌─┐ │  ┌───┐  │ ┌─┐ │   ┌───┐  │ ┌─┐ │  ┌───┐  │ ┌─┐ │
--    │ A │  │ │AB│ │  │ B │  │ │AB│ │   │ B │  │ │AB│ │  │ B │  │ │AB│ │
--    └───┘  │ └─┘ │  └───┘  │ └─┘ │   └───┘  │ └─┘ │  └───┘  │ └─┘ │
--           │─────│          │─────│           │─────│          │░░░B░│
--           └─────┘          └─────┘           └─────┘          └─────┘
```


## INNER / LEFT / RIGHT


```
// ========== INNER JOIN ==========
-- 只返回匹配的行 (最常用)
SELECT u.name, o.id AS order_id, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- 结果:
-- Alice | 1 | 100
-- Alice | 2 | 200
-- Bob   | 3 | 150

-- ========== LEFT JOIN ==========
-- 返回左表所有行,右表无匹配则 NULL
SELECT u.name, o.id AS order_id, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- 结果:
-- Alice | 1    | 100
-- Alice | 2    | 200
-- Bob   | 3    | 150
-- Carol | NULL | NULL     ← 无订单
-- Dave  | NULL | NULL     ← 无订单

-- 找无订单的用户:
SELECT u.name
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;  -- 右表为 NULL = 不匹配
-- Carol, Dave

-- ========== RIGHT JOIN ==========
-- LEFT JOIN 的反向,通常用 LEFT 替代
SELECT u.name, o.id AS order_id
FROM orders o
RIGHT JOIN users u ON u.id = o.user_id;
-- 等价于:
SELECT u.name, o.id AS order_id
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```


## 多表 JOIN


```
// ========== 三表 JOIN ==========
-- 用户 → 订单 → 订单明细
SELECT
    u.name AS user_name,
    o.id AS order_id,
    p.name AS product_name,
    oi.quantity,
    oi.price
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE u.id = 1;

-- ========== JOIN 执行顺序 ==========
-- SQL 不是从左到右执行!
-- 优化器自己决定执行顺序
-- 但逻辑上是:
-- 1. FROM + JOIN → 生成虚拟表
-- 2. WHERE → 过滤
-- 3. GROUP BY → 分组
-- 4. HAVING → 过滤分组
-- 5. SELECT → 选择列
-- 6. ORDER BY → 排序
-- 7. LIMIT → 限制

-- ========== JOIN + 聚合 ==========
-- 每个用户及其订单数和总消费
SELECT
    u.id,
    u.name,
    COUNT(o.id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_spent DESC;

-- ========== JOIN + 子查询 ==========
SELECT u.*, recent.order_date, recent.amount
FROM users u
JOIN (
    SELECT user_id, MAX(created_at) AS order_date, SUM(amount) AS amount
    FROM orders
    GROUP BY user_id
) recent ON u.id = recent.user_id;
```


## 自连接与 CROSS JOIN


```
// ========== 自连接 ==========
-- 同一张表连接自身 (必须用别名)

-- employees 表:
-- id | name  | manager_id
-- 1  | Alice | NULL       (CEO)
-- 2  | Bob   | 1
-- 3  | Carol | 1
-- 4  | Dave  | 2
-- 5  | Eve   | 2

-- 查找员工和他们的经理
SELECT
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- 结果:
-- employee | manager
-- Alice    | NULL      (CEO 无经理)
-- Bob      | Alice
-- Carol    | Alice
-- Dave     | Bob
-- Eve      | Bob

-- ========== CROSS JOIN ==========
-- 笛卡尔积: 左表每一行 × 右表每一行
-- 结果行数 = 左表行数 × 右表行数

-- 生成日期 × 产品 的组合
SELECT d.date, p.name
FROM (
    SELECT DISTINCT DATE(created_at) AS date FROM orders
) d
CROSS JOIN products p;

-- 实际用途: 生成完整的时间序列
-- 生成每个产品的每日销售 (含无销售的日期)

-- ========== NATURAL JOIN ==========
-- 自动匹配同名列 (不推荐,隐式不安全)
-- SELECT * FROM users NATURAL JOIN orders;
-- 等价于: ON users.id = orders.id (可能不是你想要的!)
```


## 完整示例: 电商查询


```
// ========== 电商查询综合 ==========
-- 1. 最近 30 天每个分类的销售排行
SELECT
    c.name AS category,
    p.name AS product,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.price) AS revenue
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
  AND o.status = 'completed'
GROUP BY c.id, c.name, p.id, p.name
ORDER BY c.name, revenue DESC;

-- 2. 复购用户 (多次下单)
SELECT u.name, COUNT(o.id) AS order_count, SUM(o.amount) AS total
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
HAVING order_count > 1
ORDER BY order_count DESC;

-- 3. 从未购买的用户 (LEFT JOIN + NULL)
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;

-- 4. 每个商品的最新订单
SELECT p.name, o.id AS order_id, o.created_at
FROM products p
JOIN LATERAL (
    SELECT oi.order_id, o.created_at
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.id
    WHERE oi.product_id = p.id
    ORDER BY o.created_at DESC
    LIMIT 1
) o ON true;
```


> **Note:** 💡 JOIN 要点: INNER 交集; LEFT 保留左表; 自连接用别名; 多表 JOIN 逐步连接; WHERE NULL 找不匹配行; CROSS JOIN 笛卡尔积谨慎使用。


## 练习


<!-- Converted from: 5_SQL JOIN连接.html -->
