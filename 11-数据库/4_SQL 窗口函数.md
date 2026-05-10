# SQL 窗口函数


## 📊 SQL 窗口函数


ROW_NUMBER/RANK/DENSE_RANK 排名、LAG/LEAD 前后行、SUM/AVG OVER 滑动窗口、NTILE 分桶、FIRST_VALUE/LAST_VALUE。


## 窗口函数基础


```
// ========== 窗口函数概念 ==========
-- 窗口函数: 在结果集的"窗口"内逐行计算
-- 与聚合函数的区别:
--   - 聚合函数: 每组返回一行
--   - 窗口函数: 每行返回一行,保留原行

-- 语法:
-- 窗口函数() OVER (
--     PARTITION BY col   -- 分区 (类似 GROUP BY)
--     ORDER BY col       -- 排序
--     ROWS/RANGE BETWEEN ...  -- 窗口框
-- )

-- ========== 数据准备 ==========
-- employees 表:
-- id | name  | department | salary
-- 1  | Alice | Engineering| 120000
-- 2  | Bob   | Engineering| 100000
-- 3  | Carol | Engineering| 110000
-- 4  | Dave  | Sales      | 80000
-- 5  | Eve   | Sales      | 90000
-- 6  | Frank | Marketing  | 85000
-- 7  | Grace | Marketing  | 95000

-- ========== 基本示例 ==========
-- 每个部门的薪资排名
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees;
-- 结果:
-- Alice | Engineering | 120000 | 1
-- Carol | Engineering | 110000 | 2
-- Bob   | Engineering | 100000 | 3
-- Eve   | Sales       | 90000  | 1
-- Dave  | Sales       | 80000  | 2
-- Grace | Marketing   | 95000  | 1
-- Frank | Marketing   | 85000  | 2
```


## 排名函数


```
// ========== 排名函数对比 ==========
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    RANK()       OVER (ORDER BY salary DESC) AS rnk,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rnk
FROM employees;

-- 相同薪资时:
-- name   | salary | row_num | rank | dense_rank
-- Alice  | 120000 | 1       | 1    | 1
-- Carol  | 110000 | 2       | 2    | 2
-- Grace  | 95000  | 3       | 3    | 3
-- Eve    | 90000  | 4       | 4    | 4
-- Frank  | 85000  | 5       | 5    | 5

-- 如果 Eve 也 95000:
-- name   | salary | row_num | rank | dense_rank
-- Alice  | 120000 | 1       | 1    | 1
-- Carol  | 110000 | 2       | 2    | 2
-- Grace  | 95000  | 3       | 3    | 3
-- Eve    | 95000  | 4       | 3    | 3  ← 并列
-- Frank  | 85000  | 5       | 5    | 4  ← 跳跃  ← 连续

-- ROW_NUMBER():   唯一连续 (同分随机)
-- RANK():         并列后跳跃
-- DENSE_RANK():   并列后连续

-- ========== 每组 Top N ==========
-- 每个部门薪资最高的 2 人
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY department ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT name, department, salary
FROM ranked
WHERE rn <= 2;
```


## LAG / LEAD 前后行


```
// ========== LAG / LEAD ==========
-- LAG: 访问前 N 行的值
-- LEAD: 访问后 N 行的值

-- 比前一个月增长
SELECT
    DATE_FORMAT(created_at, '%Y-%m') AS month,
    SUM(amount) AS revenue,
    LAG(SUM(amount)) OVER (ORDER BY DATE_FORMAT(created_at, '%Y-%m')) AS prev_month,
    SUM(amount) - LAG(SUM(amount)) OVER (ORDER BY DATE_FORMAT(created_at, '%Y-%m')) AS growth
FROM orders
GROUP BY month;

-- 每个用户当前订单和前一个订单
SELECT
    user_id,
    order_id,
    amount,
    created_at,
    LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_amount,
    COALESCE(amount - LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at), 0) AS diff
FROM orders;

-- LEAD: 下一个订单金额
SELECT
    user_id,
    amount,
    LEAD(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS next_amount,
    LEAD(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS next_order_date
FROM orders;

-- ========== 移动平均 ==========
-- 3 个月移动平均
SELECT
    month,
    revenue,
    AVG(revenue) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3m
FROM monthly_revenue;
```


## 窗口框与聚合窗口


```
// ========== 窗口框 ==========
-- 默认窗口框: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- ROWS: 基于行数
-- RANGE: 基于值范围
-- GROUPS: 基于组

-- 累计总和 (运行总计)
SELECT
    month,
    revenue,
    SUM(revenue) OVER (ORDER BY month) AS running_total
FROM monthly_revenue;

-- 3 天移动总和
SELECT
    date,
    sales,
    SUM(sales) OVER (
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_sum_3d
FROM daily_sales;

-- 未来 2 天 + 当前 (中心移动平均)
SELECT
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) AS centered_avg
FROM daily_sales;

-- 分区内的运行总计
SELECT
    user_id,
    amount,
    created_at,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY created_at
    ) AS user_running_total
FROM orders;

-- ========== 聚合窗口函数 ==========
SELECT
    id, name, department, salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    MAX(salary) OVER (PARTITION BY department) AS dept_max,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM employees;
```


## 高级窗口函数


```
// ========== NTILE ==========
-- NTILE(n): 将结果分成 n 个组 (分桶)

-- 将用户按消费分成 4 组 (四分位)
SELECT
    user_id,
    total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
FROM (
    SELECT user_id, SUM(amount) AS total_spent
    FROM orders GROUP BY user_id
) AS user_spending;

-- 前 25% 用户 = NTILE(4) = 1

-- ========== FIRST_VALUE / LAST_VALUE ==========
-- FIRST_VALUE: 窗口内第一个值
-- LAST_VALUE:  窗口内最后一个值

-- 每个部门的最高薪和最低薪
SELECT
    id, name, department, salary,
    FIRST_VALUE(name) OVER (
        PARTITION BY department ORDER BY salary DESC
    ) AS highest_paid,
    FIRST_VALUE(salary) OVER (
        PARTITION BY department ORDER BY salary DESC
    ) AS highest_salary
FROM employees;

-- ========== NTH_VALUE ==========
-- 每个部门第二高薪
SELECT *,
    NTH_VALUE(salary, 2) OVER (
        PARTITION BY department
        ORDER BY salary DESC
        RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_highest
FROM employees;

-- ========== 百分比排名 ==========
SELECT
    name, salary,
    PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank,  -- (rank-1)/(total-1)
    CUME_DIST()   OVER (ORDER BY salary) AS cumulative_dist  -- <= 当前值的占比
FROM employees;
```


> **Note:** 💡 窗口函数: OVER (PARTITION BY ... ORDER BY ...); ROW_NUMBER/RANK/DENSE_RANK 排名; LAG/LEAD 前后行对比; SUM/AVG OVER 滑动窗口; NTILE 分桶; FIRST_VALUE 每组首个。


## 练习


<!-- Converted from: 4_SQL 窗口函数.html -->
