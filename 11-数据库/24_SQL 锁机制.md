# SQL 锁机制


## 🔒 SQL 锁机制


共享锁与排他锁、意向锁、行锁/表锁/页锁、间隙锁与 Next-Key Lock、乐观锁与悲观锁、锁监控与诊断、死锁检测与预防。


## 锁的类型


```
// ========== 锁的粒度 ==========
-- 粒度从粗到细:
-- 表锁 → 页锁 → 行锁
-- 粒度越细,并发越高,开销越大

-- 1. 表锁 (Table Lock)
--    MySQL MyISAM 默认
--    锁住整张表
--    开销小,并发低

-- 2. 行锁 (Row Lock)
--    MySQL InnoDB 默认
--    锁住特定行
--    开销大,并发高

-- 3. 页锁 (Page Lock)
--    介于表锁和行锁之间
--    SQL Server 使用

-- ========== 锁的模式 (InnoDB) ==========

-- 1. 共享锁 (S Lock)
--    允许其他事务读,不允许写
--    SELECT ... LOCK IN SHARE MODE;  -- MySQL
--    SELECT ... FOR SHARE;           -- MySQL 8.0+

-- 2. 排他锁 (X Lock)
--    不允许其他事务读或写
--    SELECT ... FOR UPDATE;          -- 加排他锁
--    UPDATE / DELETE / INSERT        -- 自动加排他锁

-- 3. 意向锁 (Intention Lock)
--    表级锁,表示事务准备加行锁
--    意向共享锁 (IS): 准备加 S 锁
--    意向排他锁 (IX): 准备加 X 锁
--    意向锁不阻塞任何请求 (只阻塞整表请求)

-- ========== 锁的兼容性 ==========
--      S    X    IS   IX
-- S    ✅   ❌   ✅   ❌
-- X    ❌   ❌   ❌   ❌
-- IS   ✅   ❌   ✅   ✅
-- IX   ❌   ❌   ✅   ✅

-- S 和 S 兼容: 多个事务可以同时读
-- S 和 X 互斥: 不能同时读写同一行
-- X 和 X 互斥: 不能同时写同一行
```


## 行锁实现 (Record Lock / Gap Lock)


```
// ========== Record Lock ==========
-- 行锁: 锁住索引记录
-- InnoDB 通过索引实现行锁

-- 示例: 锁住 id=1 的行
BEGIN;
SELECT * FROM users WHERE id = 1 FOR UPDATE;
-- id=1 的行被加 X 锁 (其他事务不能修改/删除)

-- 注意: 如果没有索引 → 行锁升级为表锁!
SELECT * FROM users WHERE name = 'Alice' FOR UPDATE;
-- 如果 name 没有索引 → InnoDB 锁住所有行 (实际是表锁)

-- ========== Gap Lock (间隙锁) ==========
-- 锁住索引记录之间的"间隙"
-- 防止其他事务在间隙中插入新行
-- 只在 REPEATABLE READ 级别存在

-- 数据: id: 1, 3, 5, 7
-- 间隙: (-∞,1), (1,3), (3,5), (5,7), (7,+∞)

BEGIN;
SELECT * FROM users WHERE id BETWEEN 3 AND 5 FOR UPDATE;

-- 其他事务不能插入 id=4 的行 (间隙被锁)
-- 但可以插入 id=2 的行 (不在锁范围内)

-- ========== Next-Key Lock ==========
-- Record Lock + Gap Lock 的组合
-- InnoDB RR 级别的默认锁机制

-- 锁住: 行本身 + 之前的间隙
-- 例如 id=5 的 Next-Key Lock 锁住 (3,5]
-- 解决了幻读问题!

-- ========== 插入意向锁 ==========
-- INSERT 时的特殊间隙锁
-- 多个事务可以同时向同一间隙插入
-- (只要插入的位置不同,不冲突)
```


## 乐观锁 vs 悲观锁


```
// ========== 悲观锁 ==========
-- 假设一定会发生冲突
-- 操作数据前先加锁

-- 应用场景: 高冲突,写多读少

-- 实现:
BEGIN;
-- 查询并加锁
SELECT quantity FROM products WHERE id = 1 FOR UPDATE;

-- 检查库存
-- 如果 quantity >= 要购买的数量 → 继续
-- 否则 → ROLLBACK

-- 更新库存
UPDATE products SET quantity = quantity - 1 WHERE id = 1;

COMMIT;

-- 缺点: 并发低,易死锁

-- ========== 乐观锁 ==========
-- 假设不会发生冲突
-- 更新时检查版本

-- 应用场景: 低冲突,读多写少

-- 实现 (版本号):
-- 表结构: products(id, name, quantity, version)

-- 1. 读取数据 (无锁)
SELECT id, quantity, version FROM products WHERE id = 1;
-- → quantity=10, version=5

-- 2. 更新 (检查版本号)
UPDATE products
SET quantity = quantity - 1,
    version = version + 1
WHERE id = 1 AND version = 5;

-- 如果 version=5 匹配 → 更新成功
-- 如果 version≠5 → 更新影响 0 行 → 重试

-- ========== 乐观锁 (时间戳) ==========
-- 用 updated_at 代替 version

UPDATE products
SET quantity = quantity - 1
WHERE id = 1 AND updated_at = '2024-01-15 14:30:00';

-- ========== 选择指南 ==========
-- 悲观锁: 金融交易,库存扣减 (冲突高)
-- 乐观锁: 文章编辑,配置更新 (冲突低)
-- 分布式: Redis 分布式锁 / ZooKeeper
```


## SELECT ... FOR UPDATE 详解


```
// ========== SELECT FOR UPDATE ==========

-- 1. 基本用法
BEGIN;

-- 锁住订单 (其他事务不能修改)
SELECT * FROM orders WHERE id = 1 FOR UPDATE;

-- 执行业务逻辑...
UPDATE orders SET status = 'paid' WHERE id = 1;

COMMIT;

-- 2. NOWAIT (不等待,立即返回)
-- MySQL 8.0+:
SELECT * FROM orders WHERE id = 1 FOR UPDATE NOWAIT;
-- 如果行已被锁定 → 立即报错 (不等待)

-- 3. SKIP LOCKED (跳过已锁定的行)
SELECT * FROM orders WHERE status = 'pending'
ORDER BY created_at
LIMIT 10
FOR UPDATE SKIP LOCKED;

-- 用途: 任务队列,多个 worker 同时取任务
-- 每个 worker 取不同任务,互不干扰!

-- 4. 共享锁 (只读不阻塞读)
BEGIN;
SELECT * FROM orders WHERE id = 1 LOCK IN SHARE MODE;
-- 允许其他事务读取,不允许修改

-- 5. FOR SHARE (MySQL 8.0+ 替代 LOCK IN SHARE MODE)
SELECT * FROM orders WHERE id = 1 FOR SHARE;

-- ========== 实践建议 ==========
-- 1. 事务尽量短小
-- 2. 始终按相同顺序访问资源 (防死锁)
-- 3. 用索引减少锁范围
-- 4. 避免锁升级 (无索引 → 全表锁)
-- 5. 监控长时间持有锁的事务
```


## 锁监控与诊断


```
// ========== MySQL 锁监控 ==========

-- 1. 查看当前事务
SELECT * FROM information_schema.INNODB_TRX\G
-- trx_id, trx_state, trx_started, trx_mysql_thread_id

-- 2. 查看锁
SELECT * FROM performance_schema.data_locks\G
-- ENGINE_TRANSACTION_ID, OBJECT_NAME, LOCK_TYPE, LOCK_MODE, LOCK_STATUS

-- 3. 查看锁等待
SELECT * FROM performance_schema.data_lock_waits\G

-- 4. 查看当前进程
SHOW FULL PROCESSLIST;

-- 5. 杀死阻塞的事务
-- KILL [thread_id];

-- ========== 死锁检测 ==========
-- MySQL 自动检测死锁
-- 回滚其中一个事务 (选择回滚代价小的)

-- 查看最近一次死锁:
SHOW ENGINE INNODB STATUS\G
-- 查看 LATEST DETECTED DEADLOCK 部分

-- 死锁信息包含:
-- - 涉及的事务和 SQL
-- - 等待的资源
-- - 回滚的事务

-- ========== 减少死锁的建议 ==========
-- 1. 事务按相同顺序访问表
-- 2. 缩短事务时间
-- 3. 使用合理的隔离级别 (RC 比 RR 死锁少)
-- 4. 为表添加合适的索引
-- 5. 避免交互式事务 (等待用户输入)
-- 6. 考虑使用 NOWAIT / SKIP LOCKED
```


> **Note:** 💡 锁机制: InnoDB 行锁基于索引; 间隙锁防止幻读; Next-Key Lock=行锁+间隙锁; 悲观锁适合高冲突; 乐观锁适合低冲突; FOR UPDATE NOWAIT 不等待; SKIP LOCKED 跳过已锁行; 监控 INNODB_TRX 诊断锁问题。


## 练习


<!-- Converted from: 24_SQL 锁机制.html -->
