# SQL MERGE 与 UPSERT


## 🔄 SQL MERGE 与 UPSERT


MERGE (UPSERT) 语法、MySQL INSERT ON DUPLICATE KEY UPDATE、REPLACE INTO、PostgreSQL ON CONFLICT、MERGE 性能对比、数据同步场景。


## UPSERT 概念


```
// ========== UPSERT 概念 ==========
-- UPSERT = UPDATE + INSERT
-- 如果行存在 → UPDATE
-- 如果行不存在 → INSERT
-- 也叫: MERGE (标准 SQL 术语)

-- 使用场景:
-- 1. 数据同步 (源表 → 目标表)
-- 2. 导入数据 (存在则更新,不存在则插入)
-- 3. 计数器更新 (当天首次插入,后续累加)
-- 4. 缓存刷新

-- ========== 各数据库实现 ==========
-- MySQL:    INSERT ... ON DUPLICATE KEY UPDATE
--           REPLACE INTO
-- PostgreSQL: INSERT ... ON CONFLICT
-- SQLite:   INSERT OR REPLACE
-- Oracle:   MERGE
-- SQL 标准: MERGE

-- 示例: 用户表
-- users(id INT PK, email VARCHAR UNIQUE, name VARCHAR, login_count INT)

-- 如果 email 已存在 → 更新 name 和 login_count
-- 如果 email 不存在 → 插入新行
```


## MySQL UPSERT


```
// ========== MySQL: INSERT ON DUPLICATE KEY UPDATE ==========

-- 1. 基本用法 (基于唯一键/主键)
INSERT INTO users (id, name, email, login_count)
VALUES (1, 'Alice', 'alice@test.com', 1)
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    login_count = login_count + 1;

-- 如果 id=1 不存在 → INSERT
-- 如果 id=1 存在 → UPDATE (name='Alice', login_count+1)

-- 2. 使用唯一索引
INSERT INTO users (name, email, login_count)
VALUES ('Bob', 'bob@test.com', 1)
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    login_count = login_count + 1;
-- 如果 email 唯一索引冲突 → UPDATE

-- 3. 基于冲突列的条件更新
INSERT INTO users (id, name, email, created_at)
VALUES (1, 'Alice', 'alice@test.com', NOW())
ON DUPLICATE KEY UPDATE
    name = IF(VALUES(name) != '', VALUES(name), name),
    updated_at = NOW();

-- 4. 获取影响行数
-- 影响 1 行: INSERT (新插入)
-- 影响 2 行: UPDATE (更新) - 实际上是 DELETE+INSERT
-- 影响 0 行: 无变化

-- ========== MySQL: REPLACE ==========
-- REPLACE: 先 DELETE 再 INSERT

REPLACE INTO users (id, name, email)
VALUES (1, 'Alice', 'alice@test.com');

-- 如果 id=1 存在 → DELETE + INSERT
-- 如果 id=1 不存在 → INSERT

-- ⚠️ REPLACE 的陷阱:
-- 1. 会删除行再插入 → 触发 DELETE 触发器
-- 2. 自增 ID 会变化
-- 3. 如果表有外键 → DELETE 可能失败
-- 4. 性能比 ON DUPLICATE KEY UPDATE 差

-- 建议: 优先使用 ON DUPLICATE KEY UPDATE
```


## PostgreSQL UPSERT


```
// ========== PostgreSQL: INSERT ON CONFLICT ==========

-- 1. 基本用法
INSERT INTO users (id, name, email, login_count)
VALUES (1, 'Alice', 'alice@test.com', 1)
ON CONFLICT (id) DO UPDATE
    SET name = EXCLUDED.name,
        login_count = users.login_count + 1;

-- EXCLUDED 引用冲突时的"新值"

-- 2. 基于唯一索引冲突
INSERT INTO users (name, email, login_count)
VALUES ('Bob', 'bob@test.com', 1)
ON CONFLICT (email) DO UPDATE
    SET name = EXCLUDED.name,
        login_count = users.login_count + 1;

-- 3. ON CONFLICT DO NOTHING (跳过冲突)
INSERT INTO users (id, name, email)
VALUES (1, 'Alice', 'alice@test.com')
ON CONFLICT (id) DO NOTHING;
-- 如果已存在 → 什么也不做

-- 4. 条件冲突处理
INSERT INTO users (id, name, status)
VALUES (1, 'Alice', 'active')
ON CONFLICT (id) DO UPDATE
    SET status = EXCLUDED.status
    WHERE users.status <> 'banned';
-- 只有当前状态不是 banned 时才更新

-- ========== PostgreSQL MERGE (15+) ==========
-- PostgreSQL 15+ 支持标准 MERGE

-- MERGE INTO users AS target
-- USING (VALUES (1, 'Alice', 'alice@test.com')) AS source(id, name, email)
-- ON target.id = source.id
-- WHEN MATCHED THEN
--     UPDATE SET name = source.name, updated_at = NOW()
-- WHEN NOT MATCHED THEN
--     INSERT (id, name, email) VALUES (source.id, source.name, source.email);
```


## 标准 SQL MERGE


```
// ========== MERGE 语法 ==========
-- SQL 标准 MERGE (支持: Oracle, SQL Server, PostgreSQL 15+, MySQL 8.0?)

-- MERGE INTO target_table AS target
-- USING source_table AS source
-- ON target.key = source.key
-- WHEN MATCHED THEN
--     UPDATE SET target.col = source.col
-- WHEN NOT MATCHED THEN
--     INSERT (col1, col2) VALUES (source.col1, source.col2)
-- WHEN MATCHED AND source.flag = 'DELETE' THEN
--     DELETE;

-- Oracle 完整示例:
-- MERGE INTO products p
-- USING new_products np ON (p.id = np.id)
-- WHEN MATCHED THEN
--     UPDATE SET p.price = np.price, p.updated_at = SYSDATE
--     WHERE p.price <> np.price
-- WHEN NOT MATCHED THEN
--     INSERT (id, name, price) VALUES (np.id, np.name, np.price);

-- ========== 数据同步案例 ==========

-- 场景: 每天从外部系统同步商品数据
-- 源: staging_products (临时表,每天全量)
-- 目标: products (线上表)

-- MySQL 版本:
INSERT INTO products (id, name, price, stock, updated_at)
SELECT sp.id, sp.name, sp.price, sp.stock, NOW()
FROM staging_products sp
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    price = VALUES(price),
    stock = VALUES(stock),
    updated_at = NOW();

-- PostgreSQL 版本:
INSERT INTO products (id, name, price, stock, updated_at)
SELECT sp.id, sp.name, sp.price, sp.stock, NOW()
FROM staging_products sp
ON CONFLICT (id) DO UPDATE
    SET name = EXCLUDED.name,
        price = EXCLUDED.price,
        stock = EXCLUDED.stock,
        updated_at = NOW();
```


## 实战案例与注意事项


```
// ========== 实战 1: 计数器 ==========

-- 每天首次访问插入,后续累加
INSERT INTO daily_page_views (page_url, view_date, view_count)
VALUES ('/index', CURDATE(), 1)
ON DUPLICATE KEY UPDATE
    view_count = view_count + 1;

-- 表结构: UNIQUE KEY (page_url, view_date)

-- ========== 实战 2: 库存更新 ==========

-- 批量库存同步
INSERT INTO inventory (product_id, warehouse_id, quantity, updated_at)
VALUES
    (1, 1, 100, NOW()),
    (2, 1, 200, NOW()),
    (3, 1, 150, NOW())
ON DUPLICATE KEY UPDATE
    quantity = VALUES(quantity),
    updated_at = NOW(),
    -- 增量变化记录
    change_amount = ABS(quantity - VALUES(quantity));

-- ========== 注意事项 ==========

-- 1. 死锁风险
--    UPSERT 在并发下可能死锁
--    考虑使用事务 + 重试机制

-- 2. 自增 ID 浪费
--    MySQL ON DUPLICATE KEY UPDATE 即使 UPDATE 也会消耗自增 ID!
--    REPLACE 更是 DELETE+INSERT,消耗更大

-- 3. 触发器
--    MySQL: ON DUPLICATE KEY UPDATE 触发 UPDATE 触发器,不触发 INSERT
--    REPLACE: 触发 DELETE + INSERT 触发器

-- 4. 主从复制
--    基于语句的复制可能产生不一样的结果
--    建议使用基于行的复制 (ROW)

-- 5. 批量 UPSERT 性能
--    和批量 INSERT 一样,建议 500-1000 行一批
```


> **Note:** 💡 UPSERT: MySQL 用 ON DUPLICATE KEY UPDATE; PostgreSQL 用 ON CONFLICT; REPLACE 是先删后插 (不推荐); MERGE 是 SQL 标准语法; 注意自增 ID 消耗和死锁风险; 适合数据同步和计数器场景。


## 练习


<!-- Converted from: 28_SQL MERGE与UPSERT.html -->
