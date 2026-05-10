# SQL 死锁与诊断


## 💀 SQL 死锁与诊断


死锁产生条件、经典死锁场景、MySQL 死锁检测、SHOW ENGINE INNODB STATUS 解读、死锁日志分析、应用层死锁重试、预防策略。


## 死锁基础


```
// ========== 死锁概念 ==========
-- 死锁: 两个或多个事务互相等待对方释放锁
-- 每个事务都在等待对方占用的资源
-- 结果: 所有事务都无法继续

-- ========== 死锁四个必要条件 ==========
-- 1. 互斥: 资源一次只能被一个事务占用
-- 2. 持有并等待: 事务持有资源并等待其他资源
-- 3. 不可剥夺: 资源不能被强制剥夺
-- 4. 循环等待: 形成等待环路

-- ========== 经典死锁场景 ==========
-- 事务A:
BEGIN;
UPDATE accounts SET balance = 100 WHERE id = 1;  -- 锁住 id=1
UPDATE accounts SET balance = 200 WHERE id = 2;  -- 等待 id=2 (被B锁住)

-- 事务B:
BEGIN;
UPDATE accounts SET balance = 300 WHERE id = 2;  -- 锁住 id=2
UPDATE accounts SET balance = 400 WHERE id = 1;  -- 等待 id=1 (被A锁住)

-- 死锁! A 等 B 释放 id=2, B 等 A 释放 id=1
```


## 常见死锁场景


```
// ========== 场景 1: 不同顺序 ==========
-- 最常见的死锁原因!

-- 事务A:
UPDATE products SET price = 10 WHERE id = 1;
UPDATE products SET price = 20 WHERE id = 2;

-- 事务B:
UPDATE products SET price = 30 WHERE id = 2;
UPDATE products SET price = 40 WHERE id = 1;

-- 解决: 总是按相同顺序更新!
-- 事务A: id=1 → id=2
-- 事务B: id=1 → id=2

-- ========== 场景 2: 行锁升级 ==========
-- 无索引导致行锁变表锁

-- 表 orders, 列 order_no 无索引
-- 事务A: UPDATE orders SET ... WHERE order_no = 'A001';  -- 锁整表
-- 事务B: UPDATE orders SET ... WHERE order_no = 'B001';  -- 等待A

-- 解决: 为 order_no 加索引

-- ========== 场景 3: 插入意向锁冲突 ==========
-- 间隙锁 + 插入意向锁

-- 表: id (1, 5, 10)
-- 事务A: SELECT * FROM t WHERE id = 7 FOR UPDATE;  -- 锁住 (5,10] 间隙
-- 事务B: INSERT INTO t(id) VALUES (7);              -- 等待A释放间隙
-- 事务A: INSERT INTO t(id) VALUES (6);              -- 和B互相等待!

-- 解决: 使用 RC 隔离级别 (没有间隙锁)

-- ========== 场景 4: 外键锁 ==========
-- 子表插入/更新时检查父表 → 锁父表行

-- 事务A: INSERT INTO child(fk_id) VALUES (1);  -- 锁父表 id=1
-- 事务B: UPDATE parent SET ... WHERE id = 1;     -- 等待A
-- 事务A: UPDATE parent SET ... WHERE id = 2;     -- 等待B

-- 解决: 减少事务中的外键操作
```


## MySQL 死锁诊断


```
// ========== 查看死锁信息 ==========

-- 1. 查看最近一次死锁 (最常用!)
SHOW ENGINE INNODB STATUS\G

-- 输出解读 (LATEST DETECTED DEADLOCK 部分):
-- ------------------------
-- LATEST DETECTED DEADLOCK
-- ------------------------
-- 2024-01-15 14:30:00 0x7f1234
-- *** (1) TRANSACTION:
-- TRANSACTION 12345, ACTIVE 10 sec
-- MySQL thread id 8, OS thread handle 1234
-- UPDATE accounts SET balance = 100 WHERE id = 1
-- *** (1) WAITING FOR THIS LOCK TO BE GRANTED:
-- RECORD LOCKS space id 10 page no 3 n bits 72
-- *** (2) TRANSACTION:
-- TRANSACTION 12346, ACTIVE 5 sec
-- UPDATE accounts SET balance = 200 WHERE id = 2
-- *** (2) WAITING FOR THIS LOCK TO BE GRANTED:
-- RECORD LOCKS space id 10 page no 3 n bits 72
-- *** WE ROLL BACK TRANSACTION (1)
-- ------------------------

-- 解读要点:
-- 1. 两个事务的 SQL 是什么?
-- 2. 各自在等待什么锁?
-- 3. 哪个事务被回滚?

-- 2. 开启死锁日志
-- MySQL 5.6+:
-- SET GLOBAL innodb_print_all_deadlocks = ON;
-- 所有死锁都会记录到错误日志

-- 3. 查看事务锁等待
SELECT
    trx.trx_id,
    trx.trx_state,
    trx.trx_started,
    trx.trx_mysql_thread_id,
    trx.trx_query
FROM information_schema.INNODB_TRX trx;

-- 4. 查看锁信息
SELECT
    l.engine_transaction_id,
    l.object_schema,
    l.object_name,
    l.lock_type,
    l.lock_mode,
    l.lock_status
FROM performance_schema.data_locks l;
```


## 死锁预防策略


```
// ========== 代码层面预防 ==========

-- 1. 固定顺序访问
-- ✅ 所有事务按 id 升序更新
UPDATE accounts SET balance = ... WHERE id = 1;
UPDATE accounts SET balance = ... WHERE id = 2;

-- 2. 缩短事务
-- 不要在事务中做耗时操作 (RPC, 外部API)
BEGIN;
  UPDATE accounts SET balance = ... WHERE id = 1;
  -- ❌ 调用外部支付 API (长时间等待)
  UPDATE accounts SET balance = ... WHERE id = 2;
COMMIT;

-- ✅ 提前获取外部数据,再开启事务
-- 3. 降低隔离级别
-- RR → RC (没有间隙锁,死锁概率降低)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 4. 使用 NOWAIT / SKIP LOCKED
SELECT * FROM orders WHERE status = 'pending'
LIMIT 1 FOR UPDATE SKIP LOCKED;

-- 5. 使用索引
-- 确保 UPDATE/DELETE 的 WHERE 走索引

-- ========== 应用层重试 ==========
-- 死锁发生时,MySQL 会自动回滚一个事务
-- 应用程序应捕获死锁错误并重试

-- 死锁错误码: MySQL 1213 (ER_LOCK_DEADLOCK)

-- Python 伪代码:
-- max_retries = 3
-- retry_delay = 0.1
-- for attempt in range(max_retries):
--     try:
--         cursor.execute("BEGIN")
--         cursor.execute("UPDATE accounts ...")
--         cursor.execute("UPDATE accounts ...")
--         connection.commit()
--         break
--     except MySQLdb.Error as e:
--         if e.args[0] == 1213:  # 死锁
--             connection.rollback()
--             time.sleep(retry_delay * (2 ** attempt))  # 指数退避
--             continue
--         else:
--             raise

-- ========== 监控与告警 ==========
-- 1. 开启所有死锁日志
SET GLOBAL innodb_print_all_deadlocks = ON;

-- 2. 监控死锁计数器
-- SHOW GLOBAL STATUS LIKE '%innodb_deadlocks%';

-- 3. 设置告警: 死锁次数 > 阈值 时触发
```


> **Note:** 💡 死锁要点: 主要原因: 不同资源访问顺序; InnoDB 自动检测死锁并回滚一个事务; SHOW ENGINE INNODB STATUS 查看死锁; 应用层需捕获 1213 错误并重试; 预防: 固定顺序 + 短事务 + 索引 + RC 隔离。


## 练习


<!-- Converted from: 25_SQL 死锁与诊断.html -->
