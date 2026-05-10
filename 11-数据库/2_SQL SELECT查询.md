# SQL SELECT查询


## SQL SELECT 查询


SELECT 是 SQL 中最核心的查询语句，用于从数据库中检索数据。本章涵盖 WHERE 条件过滤、ORDER BY 排序、LIMIT/OFFSET 分页、DISTINCT 去重、AS 别名以及 CASE WHEN 条件表达式。


## SELECT 基础语法


```
// ========== 基本查询 ==========
-- 查询所有列
SELECT * FROM employees;

-- 查询指定列
SELECT id, name, salary FROM employees;

-- 查询常量表达式 (不需要表)
SELECT 1 + 1 AS result;

-- 查询时使用运算
SELECT name, salary, salary * 1.1 AS raised_salary FROM employees;

-- 使用函数
SELECT UPPER(name) AS upper_name, LENGTH(name) AS name_len FROM employees;
```


## WHERE 条件过滤


```
// ========== WHERE 基本用法 ==========
-- 比较运算符: =, !=, <>, >, <, >=, <=
SELECT name, salary FROM employees WHERE salary > 5000;
SELECT name, department FROM employees WHERE department = 'Engineering';
SELECT name, salary FROM employees WHERE salary <> 0;  -- <> 等价于 !=

// ========== AND / OR / NOT ==========
-- AND: 所有条件都满足
SELECT * FROM employees
WHERE department = 'Sales' AND salary > 6000;

-- OR: 满足任一条件
SELECT * FROM employees
WHERE department = 'Sales' OR department = 'Marketing';

-- 混合使用, AND 优先级高于 OR, 建议加括号
SELECT * FROM employees
WHERE (department = 'Sales' OR department = 'Marketing')
  AND salary > 5000;

-- NOT: 取反
SELECT * FROM employees
WHERE NOT department = 'Engineering';

-- NOT IN / NOT LIKE / NOT BETWEEN / IS NOT NULL
SELECT * FROM employees WHERE department NOT IN ('HR', 'Admin');
SELECT * FROM employees WHERE name NOT LIKE 'A%';
SELECT * FROM employees WHERE salary NOT BETWEEN 3000 AND 8000;
```


> **Note:** SQL 中字符串用单引号 '' 包裹。MySQL 可配置双引号,但标准 SQL 用单引号。AND 优先级高于 OR,混合使用时用括号明确逻辑。


## IN / BETWEEN / LIKE / IS NULL


```
// ========== IN: 匹配集合中的任意值 ==========
-- 等价于多个 OR 条件的简写
SELECT name, department FROM employees
WHERE department IN ('Engineering', 'Sales', 'Product');

-- 子查询中使用 IN
SELECT name FROM employees
WHERE id IN (SELECT DISTINCT user_id FROM orders);

// ========== BETWEEN: 范围查询 (闭区间) ==========
-- 包含边界值,等价于 salary >= 5000 AND salary <= 10000
SELECT name, salary FROM employees
WHERE salary BETWEEN 5000 AND 10000;

-- 日期范围
SELECT * FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- 注意: BETWEEN 的边界值顺序不能颠倒
-- 错误: BETWEEN 10000 AND 5000  -- 不会报错但结果为空

// ========== LIKE: 模糊匹配 ==========
-- % 匹配任意多个字符 (含零个)
-- _ 匹配单个字符

-- 以 'A' 开头的名字
SELECT name FROM employees WHERE name LIKE 'A%';

-- 包含 'son' 的名字
SELECT name FROM employees WHERE name LIKE '%son%';

-- 第二个字符为 'a' 的名字
SELECT name FROM employees WHERE name LIKE '_a%';

-- 以 'en' 结尾的名字
SELECT name FROM employees WHERE name LIKE '%en';

-- 转义: LIKE '100\%'  (MySQL 默认关闭 ESCAPE, 需 ESCAPE '\')
-- 或以 ESCAPE 自定义转义符: LIKE '100%' ESCAPE '|'

// ========== IS NULL / IS NOT NULL ==========
-- NULL 不能用 = NULL 判断,必须用 IS NULL
SELECT name FROM employees WHERE manager_id IS NULL;       -- 没上级的人(CEO)
SELECT name FROM employees WHERE email IS NOT NULL;        -- 有邮箱的人
SELECT name FROM employees WHERE phone IS NULL OR phone = ''; -- 无电话或空字符串

-- NULL 参与的运算结果都是 NULL
-- 用 COALESCE 或 IFNULL 处理
SELECT name, COALESCE(bonus, 0) AS bonus FROM employees;
```


> **Note:** NULL 不是空字符串 ''。任何值与 NULL 做比较 (=, >, <) 结果都是 UNKNOWN, 不会包含在结果中。必须用 IS NULL 或 IS NOT NULL 判断。


## ORDER BY 排序


```
// ========== ORDER BY ==========
-- ASC: 升序 (默认), DESC: 降序
SELECT name, salary FROM employees
ORDER BY salary DESC;

-- 多列排序: 先按 department 升序,再按 salary 降序
SELECT name, department, salary FROM employees
ORDER BY department ASC, salary DESC;

-- 按列位置排序 (数字代表 SELECT 中第几列)
SELECT name, salary, department FROM employees
ORDER BY 2 DESC;  -- 按第二列 (salary) 降序

-- 按表达式排序
SELECT name, salary, salary * 0.1 AS tax FROM employees
ORDER BY salary * 0.1 DESC;

-- 结合 WHERE 使用: WHERE 必须在 ORDER BY 之前
SELECT name, salary FROM employees
WHERE department = 'Engineering'
ORDER BY salary DESC;

-- NULL 的排序: ORDER BY ... NULLS LAST (PostgreSQL)
-- MySQL: NULL 默认在 ASC 时排前面, DESC 时排后面
-- 可以: ORDER BY ISNULL(name), name ASC;  -- 让 NULL 排最后
```


> **Note:** ORDER BY 必须放在 WHERE 之后。多列排序时,从左到右依次作为排序键。数值列位置方便但可读性差,建议使用列名或别名。


## LIMIT / OFFSET 分页


```
// ========== LIMIT / OFFSET (MySQL, PostgreSQL, SQLite) ==========
-- 返回前 10 条记录
SELECT * FROM employees LIMIT 10;

-- 跳过前 5 条,返回接下来的 10 条 (第 6-15 条)
SELECT * FROM employees LIMIT 10 OFFSET 5;

-- LIMIT 简写: LIMIT 偏移量, 数量
SELECT * FROM employees LIMIT 5, 10;  -- MySQL 特有语法

-- 分页公式: 每页 page_size 条, 第 page_num 页
-- LIMIT page_size OFFSET (page_num - 1) * page_size
-- 第 3 页, 每页 20 条
SELECT * FROM employees LIMIT 20 OFFSET 40;

-- 结合 ORDER BY 使用 (分页必须结合排序才有意义)
SELECT id, name, salary FROM employees
ORDER BY salary DESC
LIMIT 20 OFFSET 0;  -- 工资最高的前 20 人

// ========== TOP (SQL Server) ==========
-- SELECT TOP 10 * FROM employees;
-- SELECT TOP 10 * FROM employees ORDER BY salary DESC;

// ========== FETCH (标准 SQL / PostgreSQL / Oracle) ==========
-- SELECT * FROM employees
-- ORDER BY salary DESC
-- OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY;
```


> **Note:** 分页一定要配合 ORDER BY,否则每次查询返回的顺序可能不一致!OFFSET 越大性能越差,大数据量时可考虑基于游标的分页 (WHERE id > last_id LIMIT 20)。


## DISTINCT 去重


```
// ========== DISTINCT ==========
-- 去除重复的 department 值
SELECT DISTINCT department FROM employees;

-- DISTINCT 作用在所有 SELECT 列上,组合去重
SELECT DISTINCT department, position FROM employees;

-- 错误用法: DISTINCT 放在部分列前
-- SELECT name, DISTINCT department FROM employees;  -- 语法错误!

-- COUNT(DISTINCT ...) 统计不重复数量
SELECT
    COUNT(*) AS total,
    COUNT(DISTINCT department) AS dept_count
FROM employees;

-- DISTINCT vs GROUP BY (GROUP BY 更灵活)
SELECT department, COUNT(*) AS cnt
FROM employees
GROUP BY department;

-- DISTINCT 可用于多列组合后统计
SELECT COUNT(DISTINCT department || '-' || position) AS unique_roles
FROM employees;
-- MySQL: COUNT(DISTINCT CONCAT(department, '-', position))
```


## 别名 AS


```
// ========== 列别名 ==========
-- 简化列名或添加可读性
SELECT
    id,
    name AS employee_name,
    salary * 12 AS annual_salary,
    CASE WHEN salary > 8000 THEN '高薪' ELSE '普通' END AS salary_level
FROM employees;

-- AS 可以省略 (不推荐,容易混淆)
SELECT name employee_name FROM employees;

// ========== 表别名 ==========
-- 多表查询时简化表名
SELECT e.name, d.name AS dept_name
FROM employees AS e
JOIN departments AS d ON e.dept_id = d.id;

-- 子查询必须加别名
SELECT dept_stats.*
FROM (
    SELECT department, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
) AS dept_stats
WHERE avg_sal > 6000;

// ========== 别名使用注意 ==========
-- 别名不能在 WHERE 中使用 (执行顺序: FROM > WHERE > SELECT > ORDER BY)
-- 错误: SELECT salary * 12 AS annual FROM employees WHERE annual > 100000;
-- 正确: SELECT salary * 12 AS annual FROM employees WHERE salary * 12 > 100000;

-- 别名可以在 ORDER BY 中使用
SELECT name, salary * 12 AS annual FROM employees
ORDER BY annual DESC;
```


> **Note:** SQL 执行顺序: FROM > WHERE > GROUP BY > HAVING > SELECT > ORDER BY > LIMIT。所以 WHERE 中不能使用 SELECT 里定义的别名,但 ORDER BY 可以。


## CASE WHEN 条件表达式


```
// ========== CASE WHEN 基本语法 ==========
-- 两种写法: 简单 CASE 和 搜索 CASE

-- 搜索 CASE (最常用,支持任意条件)
SELECT
    name,
    salary,
    CASE
        WHEN salary < 3000 THEN '低薪'
        WHEN salary BETWEEN 3000 AND 8000 THEN '中薪'
        WHEN salary > 8000 THEN '高薪'
        ELSE '未知'
    END AS salary_level
FROM employees;

-- 简单 CASE (等值比较)
SELECT
    name,
    department,
    CASE department
        WHEN 'Engineering' THEN '技术部'
        WHEN 'Sales'       THEN '销售部'
        WHEN 'Marketing'   THEN '市场部'
        ELSE '其他'
    END AS dept_cn
FROM employees;

// ========== CASE WHEN 实战场景 ==========
-- 1. 数据透视 (行转列)
SELECT
    department,
    SUM(CASE WHEN gender = 'Male'   THEN 1 ELSE 0 END) AS male_count,
    SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS female_count
FROM employees
GROUP BY department;

-- 2. 条件聚合
SELECT
    department,
    COUNT(*) AS total,
    SUM(CASE WHEN salary >= 5000 THEN 1 ELSE 0 END) AS high_salary_cnt,
    ROUND(
        SUM(CASE WHEN salary >= 5000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    ) AS high_salary_pct
FROM employees
GROUP BY department;

-- 3. 自定义排序
SELECT name, department, salary FROM employees
ORDER BY
    CASE department
        WHEN 'Engineering' THEN 1
        WHEN 'Sales'       THEN 2
        WHEN 'Marketing'   THEN 3
        ELSE 4
    END;

-- 4. 更新中使用 CASE
-- UPDATE employees
-- SET salary = CASE
--     WHEN department = 'Engineering' THEN salary * 1.2
--     WHEN department = 'Sales'       THEN salary * 1.15
--     ELSE salary * 1.05
-- END;

-- 5. 标记异常数据
SELECT
    name,
    salary,
    CASE
        WHEN salary IS NULL           THEN '数据缺失'
        WHEN salary < 0              THEN '异常负数'
        WHEN salary > 100000         THEN '需人工复核'
        ELSE '正常'
    END AS salary_check
FROM employees;
```


> **Note:** CASE WHEN 是 SQL 中的"if-else",可以在 SELECT、WHERE、ORDER BY、GROUP BY 等各种子句中使用。END 关键字不能省略,ELSE 是可选的(不匹配时返回 NULL)。


## 综合示例: 员工查询


```
// ========== 综合查询示例 ==========
-- 假设表结构: employees(id, name, department, position, salary, hire_date, gender, manager_id)

-- 查询技术部工资最高的 5 名员工
SELECT
    id,
    name                                   AS employee_name,
    salary * 13 + COALESCE(bonus, 0)       AS annual_income,
    CASE
        WHEN salary >= 10000 THEN '高级'
        WHEN salary >= 6000  THEN '中级'
        ELSE '初级'
    END                                    AS level
FROM employees
WHERE department = 'Engineering'
  AND salary IS NOT NULL
ORDER BY annual_income DESC
LIMIT 5;

-- 统计各部门薪资分布
SELECT
    department,
    COUNT(*)                               AS emp_count,
    ROUND(AVG(salary), 0)                  AS avg_salary,
    MAX(salary)                            AS max_salary,
    MIN(salary)                            AS min_salary,
    ROUND(AVG(salary), 0) - MIN(salary)    AS salary_gap,
    COUNT(DISTINCT position)               AS position_count
FROM employees
WHERE salary IS NOT NULL
GROUP BY department
ORDER BY avg_salary DESC;

-- 查询入职超过 3 年的高绩效员工
SELECT
    name,
    department,
    salary,
    hire_date,
    TIMESTAMPDIFF(YEAR, hire_date, CURDATE()) AS years_served
FROM employees
WHERE hire_date <= DATE_SUB(CURDATE(), INTERVAL 3 YEAR)
  AND performance_score >= 90
  AND department IN ('Engineering', 'Product', 'Sales')
ORDER BY years_served DESC, salary DESC
LIMIT 20;
```


## 练习


<!-- Converted from: 2_SQL SELECT查询.html -->
