# MVCC实现对比 - PostgreSQL与MVCC

*深入对比 MySQL Undo Log 与 PostgreSQL 元组版本两种 MVCC 实现方式的差异，涵盖可见性规则、快照隔离及写冲突处理*

MySQL vs PostgreSQL MVCC 实现对比

| 维度 | MySQL InnoDB | PostgreSQL |
| --- | --- | --- |
| **版本存储方式** | Undo Log（单独的日志空间） | 多版本元组共存（同一表空间内） |
| **UPDATE 行为** | 原地修改 + 旧值写入 undo log | 创建新元组 + 旧行标记 xmax |
| **版本链方向** | 从当前记录向下（roll_pointer） | 从旧行到新行（ctid 指针） |
| **空间回收** | Purge 线程清理 undo log | VACUUM 回收死元组 |
| **聚簇索引** | 有（数据按主键聚集） | 无堆表（Heap），索引和数据分离 |
| **锁机制** | 有间隙锁（RR 级别） | 无间隙锁（用 Serializable Snapshot） |
| **回滚支持** | undo log 天然支持 | 需要显式 ROLLBACK 释放新元组 |

InnoDB 基于 Read View 的可见性

InnoDB 在事务开始时（RR）或每条 SQL 开始时（RC）创建 Read View，包含活跃事务列表（m_ids）。对某行数据的可见性：


1. trx_id == 自身事务 ID →
   可见
   （自己修改的数据）
2. trx_id < m_up_limit_id →
   可见
   （创建 Read View 前已提交）
3. trx_id >= m_low_limit_id →
   不可见
   （Read View 创建后才开始的事务）
4. trx_id 在 m_ids 中 →
   不可见
   （未提交的活跃事务）
5. trx_id 不在 m_ids 中且在范围内 →
   可见
   （已提交事务）


如果当前版本不可见，沿 roll_pointer 回溯 undo log 找更旧的可见版本。

PostgreSQL 基于 xmin/xmax 的可见性

PostgreSQL 在事务开始时创建快照（Snapshot），包含当前活跃事务 ID 列表。对某个元组：


1. **检查 xmin（创建者）：**

   - xmin == 自身事务 ID 且 cmin < 当前命令序号 →
      可见
   - xmin 已提交且在快照创建之前 →
      可见
   - 其他情况 →
      不可见
      （创建者未提交或在快照后）
2. **检查 xmax（删除者/更新者）：**

   - xmax == 0 →
      未被删除
   - xmax 已提交 →
      已被删除/更新
   - xmax 未提交 →
      仍可见
      （可能回滚）


关键差异：PG 不需要回溯版本链，直接通过元组自身的 xmin/xmax 判断可见性。

RR vs RC 快照差异

| 隔离级别 | MySQL InnoDB | PostgreSQL |
| --- | --- | --- |
| REPEATABLE READ | 首次 SELECT 时创建 Read View，整个事务复用 | 首次 SELECT 时创建 Snapshot，整个事务复用 |
| READ COMMITTED | 每条 SELECT 重新创建 Read View | 每条 SELECT 重新创建 Snapshot |
| SERIALIZABLE | 加锁读（S/X 锁 + 间隙锁） | SSI（Serializable Snapshot Isolation） |

InnoDB 写冲突处理

```
-- InnoDB 使用锁机制处理写冲突
-- 事务 A                           -- 事务 B
BEGIN;                              BEGIN;
UPDATE users SET name='A'           UPDATE users SET name='B'
  WHERE id = 1;                      WHERE id = 1;
-- 获取 id=1 的 X 锁，执行更新      -- 等待事务 A 释放 X 锁...

-- 两种可能的结果：
-- 1. 事务 A 提交/回滚后，事务 B 获取锁继续执行
-- 2. 超过 innodb_lock_wait_timeout（默认50秒），事务 B 报错:
--    ERROR 1205: Lock wait timeout exceeded

-- 如果存在死锁，InnoDB 自动回滚其中一个事务:
-- ERROR 1213: Deadlock found when trying to get lock
```

PostgreSQL 写冲突处理（First-Committer-Wins）

```
-- PostgreSQL 使用乐观并发控制处理写冲突
-- 不阻塞，立即检测冲突

-- 事务 A                           -- 事务 B
BEGIN;                              BEGIN;
SELECT * FROM users WHERE id=1;     SELECT * FROM users WHERE id=1;
-- 看到 xmin=50 的版本               -- 也看到 xmin=50 的版本

UPDATE users SET name='A'           UPDATE users SET name='B'
  WHERE id = 1;                      WHERE id = 1;
-- 创建新元组 xmin=100, 旧行 xmax=100  -- 尝试更新，检测到 xmax != 0
-- 成功                                -- 错误: could not serialize access
                                    --        due to concurrent update

COMMIT;                             ROLLBACK;  -- 事务 B 必须重试

-- First-Committer-Wins 规则：
-- 后提交的事务发现数据已被其他事务修改时，立即报错回滚
-- 不等待锁，不阻塞，性能更高
-- 但应用需要处理 serialization failure 并重试
```

MySQL Purge vs PostgreSQL VACUUM

| 维度 | MySQL InnoDB Purge | PostgreSQL VACUUM |
| --- | --- | --- |
| 清理对象 | Undo log 中不再需要的旧版本 | 堆表中的死元组（dead tuples） |
| 触发方式 | 后台 purge 线程自动执行 | autovacuum 自动 + 手动 VACUUM |
| 空间膨胀 | undo log 空间可能膨胀（长事务阻止 purge） | 表和索引可能膨胀（需 VACUUM FULL） |
| 在线清理 | 不阻塞读写 | VACUUM 不阻塞，VACUUM FULL 阻塞 |
| 回卷风险 | 无（undo log 是循环复用的） | 有（事务 ID 32 位，需冻结机制） |

基于 MVCC 差异的数据库选型考虑

| 场景 | 推荐 | 原因 |
| --- | --- | --- |
| 高并发 OLTP（大量短事务） | MySQL | 锁竞争可控，undo log 空间复用 |
| 复杂分析查询 | PostgreSQL | 窗口函数强大，无间隙锁干扰 |
| 频繁 UPDATE 的表 | MySQL | 原地更新，不会产生大量死元组 |
| 读多写少的场景 | PostgreSQL | MVCC 开销低，读完全不加锁 |
| 需要严格可序列化 | PostgreSQL | SSI 不加锁即可实现，性能更好 |
| JSON/文档存储需求 | PostgreSQL | JSONB 类型强大，Gin 索引高效 |
| 已有 MySQL 生态 | MySQL | 运维成熟，工具链丰富 |


<!-- Converted from: 02_MVCC实现对比.html -->
