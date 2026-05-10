# SQL 备份与恢复


## 💾 SQL 备份与恢复


mysqldump 逻辑备份、物理备份 (XtraBackup)、PostgreSQL pg_dump/pg_restore、备份策略 (全量/增量/差异)、时间点恢复 PITR、自动化备份脚本。


## 备份基础


```
// ========== 备份为什么重要 ==========
-- 备份是数据安全的最后一道防线!
-- 常见数据丢失原因:
-- 1. 人为误操作 (DELETE 忘带 WHERE)
-- 2. 硬件故障 (磁盘损坏)
-- 3. 软件 Bug (升级后数据损坏)
-- 4. 安全攻击 (勒索病毒)
-- 5. 自然灾害

-- ========== 备份类型 ==========

-- 1. 逻辑备份
--   导出 SQL 语句,可跨版本/跨平台
--   工具: mysqldump, pg_dump
--   优点: 灵活,可部分恢复
--   缺点: 慢,文件大

-- 2. 物理备份
--   直接复制数据文件
--   工具: XtraBackup (MySQL), pg_basebackup (PG)
--   优点: 快,文件小
--   缺点: 依赖版本,不可跨平台

-- 3. 全量备份
--   备份全部数据

-- 4. 增量备份
--   只备份上次备份后的变化

-- 5. 差异备份
--   只备份上次全量备份后的变化

-- ========== 备份策略 ==========
-- 常见策略:
-- 每天全量备份 (小数据量)
-- 每周全量 + 每天增量 (大数据量)
-- 主从复制作为实时备份

-- 恢复目标:
-- RPO (Recovery Point Objective): 允许丢失多少数据
-- RTO (Recovery Time Objective):  需要多久恢复
```


## mysqldump 逻辑备份


```
// ========== mysqldump ==========
-- MySQL 最常用的逻辑备份工具

-- 1. 备份单个数据库
mysqldump -u root -p mydb > mydb_backup.sql

-- 2. 备份多个数据库
mysqldump -u root -p --databases db1 db2 > multi_db.sql

-- 3. 备份所有数据库
mysqldump -u root -p --all-databases > all_dbs.sql

-- 4. 只备份表结构 (不备份数据)
mysqldump -u root -p --no-data mydb > mydb_schema.sql

-- 5. 只备份数据 (不备份结构)
mysqldump -u root -p --no-create-info mydb > mydb_data.sql

-- 6. 备份特定表
mysqldump -u root -p mydb users orders > users_orders.sql

-- 7. 常用参数
mysqldump -u root -p \
    --single-transaction \    # InnoDB 一致性备份,不加锁
    --routines \              # 包含存储过程
    --triggers \              # 包含触发器
    --events \                # 包含事件
    --set-gtid-purged=OFF \   # GTID 复制相关
    --hex-blob \              # 二进制数据用十六进制
    mydb > backup.sql

-- 8. 压缩备份 (节省空间)
mysqldump -u root -p mydb | gzip > mydb_backup.sql.gz

-- 9. 带时间戳的备份
mysqldump -u root -p mydb > mydb_$(date +%Y%m%d_%H%M%S).sql

-- ========== 恢复 ==========
-- 恢复整个数据库
mysql -u root -p mydb < mydb_backup.sql

-- 从压缩文件恢复
gunzip < mydb_backup.sql.gz | mysql -u root -p mydb

-- 恢复单个表
mysql -u root -p mydb -e "source /path/to/table_backup.sql"

-- 注意:
-- 1. 恢复前确保目标数据库存在
-- 2. 大文件恢复用 source 命令比管道更快
-- 3. 先 SET autocommit=0; 加速恢复
```


## PostgreSQL 备份与恢复


```
// ========== pg_dump (逻辑备份) ==========

-- 1. 备份单个数据库
pg_dump -U postgres mydb > mydb_backup.sql

-- 2. 自定义格式 (压缩,并行,灵活)
pg_dump -U postgres -Fc mydb > mydb_backup.dump

-- 3. 只备份结构
pg_dump -U postgres --schema-only mydb > schema.sql

-- 4. 只备份数据
pg_dump -U postgres --data-only mydb > data.sql

-- 5. 压缩
pg_dump -U postgres mydb | gzip > mydb_backup.sql.gz

-- ========== pg_restore (恢复) ==========

-- 1. 从 SQL 文件恢复
psql -U postgres mydb < mydb_backup.sql

-- 2. 从自定义格式恢复
pg_restore -U postgres -d mydb mydb_backup.dump

-- 3. 并行恢复 (4 个线程)
pg_restore -U postgres -d mydb -j 4 mydb_backup.dump

-- 4. 恢复前先创建数据库
-- createdb -U postgres mydb_new

-- ========== pg_basebackup (物理备份) ==========
-- 创建物理备份,用于 PITR

-- pg_basebackup -U postgres -D /backup/pg_base \
--     -Fp -Xs -P -v

-- 参数:
-- -Fp:  plain 格式 (文件复制)
-- -Xs:  include WAL (事务日志)
-- -P:   显示进度
-- -v:   详细输出
```


## 时间点恢复 (PITR)


```
// ========== MySQL PITR ==========
-- 需要: binlog (二进制日志)
-- 原理: 全量备份 + binlog 重放

-- 1. 启用 binlog (my.cnf)
-- [mysqld]
-- log_bin = /var/log/mysql/mysql-bin.log
-- expire_logs_days = 7
-- max_binlog_size = 100M

-- 2. 全量备份时记录 binlog 位置
mysqldump -u root -p \
    --single-transaction \
    --master-data=2 \
    mydb > mydb_backup.sql
-- --master-data=2 在备份文件中记录 binlog 文件名和位置

-- 3. 恢复到指定时间点
-- 先恢复全量备份
mysql -u root -p mydb < mydb_backup.sql

-- 再重放 binlog 到指定时间
mysqlbinlog /var/log/mysql/mysql-bin.000012 \
    --stop-datetime="2024-01-15 14:30:00" \
    | mysql -u root -p mydb

-- 恢复 binlog (跳过误操作)
mysqlbinlog /var/log/mysql/mysql-bin.000012 \
    --stop-datetime="2024-01-15 14:29:00" \
    > recover_before.sql

mysqlbinlog /var/log/mysql/mysql-bin.000012 \
    --start-datetime="2024-01-15 14:31:00" \
    > recover_after.sql

-- 先应用全量,再应用 binlog
mysql -u root -p mydb < mydb_backup.sql
mysql -u root -p mydb < recover_before.sql
-- 跳过了 14:29-14:31 的误操作!

-- ========== PostgreSQL PITR ==========
-- PostgreSQL 通过 WAL (Write-Ahead Log) 实现

-- 1. 配置 WAL 归档 (postgresql.conf)
-- wal_level = replica
-- archive_mode = on
-- archive_command = 'cp %p /backup/wal/%f'

-- 2. 基础备份
-- pg_basebackup -U postgres -D /backup/base -Fp -Xs -P

-- 3. 恢复到指定时间点
-- 创建 recovery.conf:
-- restore_command = 'cp /backup/wal/%f %p'
-- recovery_target_time = '2024-01-15 14:30:00'

-- 4. 启动 PostgreSQL 自动恢复
```


## 自动化备份脚本


```
// ========== MySQL 自动备份脚本 ==========
#!/bin/bash
# 每天凌晨 2 点执行: 0 2 * * * /backup/scripts/mysql_backup.sh

BACKUP_DIR="/backup/mysql"
DB_NAME="mydb"
USER="root"
PASSWORD="your_password"
DATE=$(date +%Y%m%d_%H%M%S)
KEEP_DAYS=7

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份 (压缩)
mysqldump -u $USER -p$PASSWORD \
    --single-transaction \
    --routines \
    --triggers \
    --events \
    $DB_NAME | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz

# 删除 7 天前的备份
find $BACKUP_DIR -name "${DB_NAME}_*.sql.gz" -mtime +$KEEP_DAYS -delete

# 记录日志
echo "[$(date)] Backup completed: ${DB_NAME}_${DATE}.sql.gz" >> $BACKUP_DIR/backup.log

# ========== 备份策略建议 ==========
-- 小项目 (数据 < 10GB):
--   每天 mysqldump
--   保留 7-30 天

-- 中项目 (10-100GB):
--   每天物理备份 (XtraBackup)
--   开启 binlog (PITR 恢复)
--   保留 7 天全量 + 30 天 binlog

-- 大项目 (> 100GB):
--   每周全量 + 每天增量
--   主从复制 (实时备份)
--   跨机房备份
--   定期演练恢复

-- ========== 恢复演练 ==========
-- 定期测试备份是否可用!
-- 至少每月一次恢复演练

-- 演练步骤:
-- 1. 在测试服务器上恢复备份
-- 2. 验证数据完整性 (行数/金额/最新时间)
-- 3. 记录恢复时间 (RTO)
-- 4. 修复发现的问题

-- "没有验证过的备份 = 没有备份"
```


> **Note:** 💡 备份要点: mysqldump 逻辑备份灵活; 物理备份快; binlog/WAL 实现 PITR; 全量+增量结合; 定期恢复演练; 3-2-1 原则: 3 份数据,2 种介质,1 份异地; 没有验证过的备份等于没有备份。


## 练习


<!-- Converted from: 16_SQL 备份与恢复.html -->
