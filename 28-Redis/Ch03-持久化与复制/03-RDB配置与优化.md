# RDB配置与优化

## 一、概念说明

合理配置RDB参数可以平衡数据安全性和性能。需要根据业务数据变化频率和可接受的数据丢失范围来调整save规则。

## 二、具体配置

### save规则配置

```bash
# 高安全场景（数据变化频繁）
save 300 10      # 5分钟内10次变化
save 60 1000     # 1分钟内1000次变化
save 10 10000    # 10秒内10000次变化

# 一般场景
save 900 1       # 15分钟内1次变化
save 300 10      # 5分钟内10次变化
save 60 10000    # 1分钟内10000次变化

# 低安全场景（数据变化不频繁）
save 3600 1      # 1小时内1次变化

# 禁用RDB
save ""
# 或在redis.conf中注释所有save行
```

### fork优化

```bash
# 避免fork阻塞
# 1. 使用大页内存（Transparent Huge Pages）
#    建议关闭：echo never > /sys/kernel/mm/transparent_hugepage/enabled

# 2. 控制Redis最大内存
#    建议不超过物理内存的70%
maxmemory 8gb

# 3. 监控fork耗时
INFO persistence | grep rdb_last_bgsave_time_sec
# 输出: rdb_last_bgsave_time_sec:1
# 超过1秒需要关注
```

### 磁盘IO优化

```bash
# 使用SSD硬盘
# RDB文件写入速度显著提升

# 独立磁盘分区
# 将RDB文件放到独立的磁盘分区
dir /data/redis

# 避免磁盘满载
# 监控磁盘使用率
df -h /data/redis

# 设置RDB文件大小限制
# redis本身不支持，需要外部脚本监控
```

### RDB文件管理

```bash
# 设置RDB文件名
dbfilename dump.rdb

# 设置保存目录
dir /var/lib/redis

# 备份RDB文件
cp /var/lib/redis/dump.rdb /backup/redis/dump-$(date +%Y%m%d).rdb

# 定时备份脚本
# crontab -e
0 2 * * * cp /var/lib/redis/dump.rdb /backup/redis/dump-$(date +\%Y\%m\%d).rdb
```

## 三、监控指标

```bash
# 查看RDB状态
INFO persistence

# 关键指标
rdb_last_bgsave_status:ok           # 最后一次保存状态
rdb_last_bgsave_time_sec:1          # 最后一次保存耗时
rdb_changes_since_last_save:100     # 上次保存后的变化数
rdb_bgsave_in_progress:0            # 是否正在保存
rdb_last_cow_size:1048576           # COW使用的内存
```

## 四、注意事项与常见陷阱

1. **不要频繁触发BGSAVE**：频繁fork影响性能
2. **监控fork时间**：超过1秒应优化
3. **COW内存**：fork期间大量写操作会占用额外内存
4. **磁盘空间**：确保RDB文件目录有足够空间
5. **并行BGSAVE**：不能同时运行多个BGSAVE
6. **关闭THP**：透明大页会增加fork阻塞时间
