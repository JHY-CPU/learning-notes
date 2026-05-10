# SQL 正则表达式


## 🔤 SQL 正则表达式


MySQL REGEXP 操作符、正则模式语法、REGEXP_REPLACE/REGEXP_SUBSTR/REGEXP_INSTR、PostgreSQL ~ 操作符、正则提取与替换、数据清洗案例。


## MySQL 正则基础


```
// ========== MySQL REGEXP ==========
-- MySQL 8.0+ 使用 ICU 正则引擎

-- 1. REGEXP 操作符 (返回 0/1)
SELECT 'hello' REGEXP '^h';     -- 1 (匹配)
SELECT 'hello' REGEXP '^x';     -- 0 (不匹配)

-- 2. NOT REGEXP
SELECT 'hello' NOT REGEXP '^x';  -- 1

-- 3. REGEXP_LIKE() 等效
SELECT REGEXP_LIKE('hello', '^h');  -- 1

-- ========== 常用模式 ==========

-- 基本匹配
SELECT * FROM users WHERE email REGEXP '@gmail\.com$';  -- Gmail 用户
SELECT * FROM users WHERE phone REGEXP '^1[3-9]';        -- 手机号开头
SELECT * FROM users WHERE name REGEXP '^[A-Z]';          -- 大写字母开头

-- 字符类
-- [abc]:   a, b, 或 c
-- [^abc]:  不是 a, b, c
-- [a-z]:   小写字母
-- [0-9]:   数字

-- 预定义类
-- [:digit:]  数字
-- [:alpha:]  字母
-- [:space:]  空白
-- [:upper:]  大写
-- [:lower:]  小写

-- 数量词
-- *:     0 次或多次
-- +:     1 次或多次
-- ?:     0 次或 1 次
-- {n}:   恰好 n 次
-- {n,}:  至少 n 次
-- {n,m}: n 到 m 次

-- 位置
-- ^:     字符串开始
-- $:     字符串结束
-- \\b:   单词边界 (MySQL 用 [[:<:]] / [[:>:]])
```


## MySQL 正则函数


```
// ========== REGEXP_REPLACE ==========
-- 正则替换

-- 1. 去除所有数字
SELECT REGEXP_REPLACE('abc123def456', '[0-9]', '');
-- 'abcdef'

-- 2. 格式化手机号
SELECT REGEXP_REPLACE('13800138000', '(\\d{3})(\\d{4})(\\d{4})', '$1****$3');
-- '138****8000'

-- 3. 清理 HTML 标签
SELECT REGEXP_REPLACE('Hello', '<[^>]+>', '');
-- 'Hello'

-- 4. 合并空格
SELECT REGEXP_REPLACE('Hello   World  SQL', '\\s+', ' ');
-- 'Hello World SQL'

-- ========== REGEXP_SUBSTR ==========
-- 提取匹配的子串

-- 1. 提取邮箱
SELECT REGEXP_SUBSTR('Contact: alice@test.com', '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}');
-- 'alice@test.com'

-- 2. 提取数字
SELECT REGEXP_SUBSTR('Order #12345', '\\d+');
-- '12345'

-- ========== REGEXP_INSTR ==========
-- 返回匹配位置

SELECT REGEXP_INSTR('hello world', 'world');     -- 7
SELECT REGEXP_INSTR('hello 123', '[0-9]');        -- 7

-- ========== 完整验证示例 ==========
-- 邮箱验证
SELECT * FROM users
WHERE email REGEXP '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$';

-- 手机号验证
SELECT * FROM users
WHERE phone REGEXP '^1[3-9]\\d{9}$';
```


## PostgreSQL 正则


```
// ========== PostgreSQL 正则 ==========
-- PostgreSQL 正则更强大 (类似 Perl)

-- 1. ~ 匹配 (大小写敏感)
SELECT 'hello' ~ '^h';      -- true
SELECT 'hello' ~ '^x';      -- false

-- 2. ~* 匹配 (大小写不敏感)
SELECT 'Hello' ~* '^h';     -- true

-- 3. !~ 不匹配
SELECT 'hello' !~ '^x';     -- true

-- 4. !~* 不匹配 (不区分大小写)
SELECT 'Hello' !~* '^x';    -- true

-- ========== 正则函数 ==========

-- 1. REGEXP_MATCHES (返回数组)
SELECT REGEXP_MATCHES('hello123world456', '\d+', 'g');
-- {123,456}

-- 2. REGEXP_REPLACE
SELECT REGEXP_REPLACE('Hello 123 World', '\d+', '***', 'g');
-- 'Hello *** World'
-- 'g' = 全局替换

-- 3. REGEXP_SPLIT_TO_TABLE
SELECT REGEXP_SPLIT_TO_TABLE('a,b,c', ',');
-- a
-- b
-- c

-- 4. REGEXP_SPLIT_TO_ARRAY
SELECT REGEXP_SPLIT_TO_ARRAY('a,b,c', ',');
-- {a,b,c}

-- ========== 高级模式 ==========
-- 提取所有邮箱
-- SELECT REGEXP_MATCHES(
--     'Contact: alice@test.com, CC: bob@test.com',
--     '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
--     'g'
-- );

-- 正则替换引用
-- SELECT REGEXP_REPLACE(
--     'hello world',
--     '(\w+) (\w+)',
--     '\2 \1'  -- 交换顺序
-- );
-- 'world hello'
```


## 正则实战: 数据清洗


```
// ========== 数据清洗案例 ==========

-- 创建脏数据表
CREATE TABLE dirty_data (
    id INT PRIMARY KEY,
    raw_phone VARCHAR(50),      -- 格式混乱
    raw_email VARCHAR(100),     -- 含有无效字符
    raw_name VARCHAR(100)       -- 多余空格/符号
);

INSERT INTO dirty_data VALUES
(1, '138-0013-8000', 'Alice Smith ', '  Alice!! Smith  '),
(2, '139.0013.8001', '"Bob" Jones (bob@test.com)', '***BOB*** jones'),
(3, '140-0013.8002', 'carol@test.com', '  Carol   Brown  ');

-- 1. 清洗手机号 (只留数字)
SELECT
    raw_phone,
    REGEXP_REPLACE(raw_phone, '[^0-9]', '') AS clean_phone
FROM dirty_data;

-- 2. 提取邮箱
SELECT
    raw_email,
    REGEXP_SUBSTR(raw_email, '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}') AS clean_email
FROM dirty_data;

-- 3. 清洗姓名
SELECT
    raw_name,
    REGEXP_REPLACE(REGEXP_REPLACE(
        TRIM(raw_name),
        '[^a-zA-Z\\s]', ''      -- 去除非字母字符
    ), '\\s+', ' ')             -- 合并空格
    AS clean_name
FROM dirty_data;

-- ========== 正则性能注意事项 ==========

-- 1. 正则查询不能使用索引!
--    大数据量上正则 = 全表扫描

-- 2. 简单 LIKE 能用索引 (前缀匹配)
SELECT * FROM users WHERE email LIKE 'alice%';  -- 走索引
SELECT * FROM users WHERE email REGEXP '^alice'; -- 不走索引

-- 3. 尽量用简单函数替代正则
SELECT * FROM users WHERE LEFT(email, 5) = 'alice'; -- 可能走索引

-- 4. 大量数据清洗建议用 ETL 工具
--    在数据库外处理 (Python pandas, Spark)
```


> **Note:** 💡 正则需要: REGEXP 匹配; REGEXP_REPLACE 替换; REGEXP_SUBSTR 提取; PostgreSQL ~ 操作符; 正则不能走索引; 数据清洗推荐 REGEXP_REPLACE; 大量数据清洗用外部工具; 注意转义符 \\ 的使用。


## 练习


<!-- Converted from: 36_SQL 正则表达式.html -->
