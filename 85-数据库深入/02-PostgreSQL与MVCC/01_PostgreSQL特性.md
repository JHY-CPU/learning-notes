# PostgreSQL特性 - PostgreSQL与MVCC

*深入理解 MVCC 元组版本机制、VACUUM、CTE、窗口函数、JSONB 数据类型及扩展生态（PostGIS/pg_trgm）*

PostgreSQL 元组可见性判断

1. 如果
   `xmin`
   = 当前事务 ID：自己创建的，
   可见
2. 如果
   `xmin`
   对应事务已提交，且
   `xmin`
   在快照创建之前：创建者已提交，
   可见
3. 如果
   `xmin`
   对应事务未提交，或在快照之后开始：创建者未提交，
   不可见
4. 对于可见的元组，检查
   `xmax`
   ：

   - `xmax`
      = 0：未被删除/更新，
      有效
   - `xmax`
      对应事务已提交：已被删除/更新，
      不可见
   - `xmax`
      对应事务未提交：正在被删除/更新，
      仍可见
      （回滚则恢复）

VACUUM 解决的三个问题

| 问题 | 说明 | 后果 |
| --- | --- | --- |
| 空间膨胀 | 死元组占据磁盘空间不释放 | 表和索引体积持续增长 |
| 事务 ID 回卷 | 事务 ID 是 32 位无符号整数，约 42 亿后会回卷 | 旧数据可能"消失" |
| 可见性映射过期 | 影响 Index-Only Scan 的效率 | 不必要的堆访问 |

JSON vs JSONB 对比

| 特性 | JSON | JSONB |
| --- | --- | --- |
| 存储格式 | 原始文本 | 分解后的二进制 |
| 写入速度 | 更快（无需解析） | 稍慢（需解析） |
| 查询速度 | 每次需解析 | 更快（已解析） |
| 索引支持 | 不支持 | Gin 索引 |
| 保留键顺序 | 是 | 否 |
| 保留重复键 | 是（取最后一个） | 否 |


<!-- Converted from: 01_PostgreSQL特性.html -->

## VACUUM 详解

### VACUUM vs VACUUM FULL

| 特性 | VACUUM | VACUUM FULL |
| --- | --- | --- |
| 锁级别 | 轻量级锁（不阻塞读写） | 排它锁（阻塞所有操作） |
| 空间回收 | 标记为可用但不还给 OS | 重写表并还给 OS |
| 速度 | 快 | 慢（需重写整个表） |
| 使用场景 | 日常维护 | 表严重膨胀时 |

### autovacuum 配置建议

```sql
-- 推荐的 autovacuum 配置
ALTER TABLE large_table SET (
  autovacuum_vacuum_threshold = 50,       -- 至少 50 个死元组才触发
  autovacuum_vacuum_scale_factor = 0.01,  -- 或 1% 的行变化
  autovacuum_analyze_threshold = 50,
  autovacuum_analyze_scale_factor = 0.005 -- 0.5% 变化时收集统计
);
```

## CTE（公用表表达式）

```sql
-- 递归 CTE：生成层次结构
WITH RECURSIVE org_tree AS (
  -- 基础情况
  SELECT id, name, manager_id, 1 AS depth
  FROM employees WHERE manager_id IS NULL
  UNION ALL
  -- 递归情况
  SELECT e.id, e.name, e.manager_id, t.depth + 1
  FROM employees e JOIN org_tree t ON e.manager_id = t.id
)
SELECT * FROM org_tree;

-- CTE 优化：MATERIALIZED 提示
-- PostgreSQL 12+ CTE 可能被内联，MATERIALIZED 强制物化
WITH expensive_calc AS MATERIALIZED (
  SELECT * FROM large_table WHERE complex_condition
)
SELECT * FROM expensive_calc a JOIN expensive_calc b ON a.id = b.ref_id;
```

## 窗口函数

```sql
-- 常用窗口函数
SELECT
  name,
  department,
  salary,
  -- 排名
  ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank,
  -- 部门平均薪资
  AVG(salary) OVER (PARTITION BY department) AS dept_avg,
  -- 累计求和
  SUM(salary) OVER (ORDER BY salary) AS running_total,
  -- 前一行和后一行
  LAG(salary, 1) OVER (ORDER BY salary) AS prev_salary,
  LEAD(salary, 1) OVER (ORDER BY salary) AS next_salary
FROM employees;
```

## PostgreSQL 扩展生态

| 扩展 | 功能 | 典型场景 |
| --- | --- | --- |
| PostGIS | 空间数据处理 | 地理信息系统、地图服务 |
| pg_trgm | 模糊文本搜索 | 模糊匹配、相似度查询 |
| pg_stat_statements | SQL 性能分析 | 慢查询排查 |
| TimescaleDB | 时序数据 | IoT、监控数据 |
| pgvector | 向量相似度搜索 | AI 嵌入、语义搜索 |
| pg_cron | 定时任务 | 数据库内定时执行 |

```sql
-- pg_trgm 模糊搜索示例
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_name_trgm ON users USING gin (name gin_trgm_ops);
SELECT name, similarity(name, '张三') AS sim
FROM users WHERE name % '张三' ORDER BY sim DESC LIMIT 10;
```
