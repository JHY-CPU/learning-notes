# 阿里云RDS

## 一、概念说明

阿里云RDS是托管关系型数据库服务，支持MySQL、PostgreSQL、SQL Server和MariaDB引擎。提供高可用、备份恢复、性能优化等功能。

| 版本 | 特点 | 适用场景 |
|------|------|----------|
| 基础版 | 单节点，成本低 | 开发测试 |
| 高可用版 | 主备架构 | 生产环境 |
| 集群版 | 读写分离 | 高并发读 |
| 三节点企业版 | 金融级高可用 | 金融核心 |

## 二、具体用法

### 创建实例

```bash
# 创建RDS MySQL实例
aliyun rds CreateDBInstance \
    --RegionId cn-hangzhou \
    --Engine MySQL \
    --EngineVersion 8.0 \
    --DBInstanceClass rds.mysql.s2.large \
    --DBInstanceStorage 100 \
    --StorageType cloud_essd \
    --DBInstanceNetType Intranet \
    --PayType Postpaid \
    --VPCId vpc-bp1xxxxxxxx \
    --VSwitchId vsw-bp1xxxxxxxx \
    --SecurityIPList 10.0.0.0/16 \
    --DBInstanceDescription production-db

# 创建数据库账号
aliyun rds CreateAccount \
    --DBInstanceId rm-bp1xxxxxxxx \
    --AccountName app_user \
    --AccountPassword "StrongDBPass123!" \
    --AccountType Normal

# 创建数据库
aliyun rds CreateDatabase \
    --DBInstanceId rm-bp1xxxxxxxx \
    --DBName myapp \
    --CharacterSetName utf8mb4

# 授权
aliyun rds GrantAccountPrivilege \
    --DBInstanceId rm-bp1xxxxxxxx \
    --AccountName app_user \
    --DBName myapp \
    --AccountPrivilege ReadWrite
```

### 连接和管理

```bash
# 获取连接地址
aliyun rds DescribeDBInstanceAttribute \
    --DBInstanceId rm-bp1xxxxxxxx

# 连接数据库
mysql -h rm-bp1xxxxxxxx.mysql.rds.aliyuncs.com -u app_user -p myapp

# 创建只读副本
aliyun rds CreateReadOnlyDBInstance \
    --RegionId cn-hangzhou \
    --MasterInstanceId rm-bp1xxxxxxxx \
    --Engine MySQL \
    --EngineVersion 8.0 \
    --DBInstanceClass rds.mysql.s2.large \
    --DBInstanceStorage 100 \
    --PayType Postpaid
```

### 备份恢复

```bash
# 手动备份
aliyun rds CreateBackup \
    --DBInstanceId rm-bp1xxxxxxxx \
    --BackupMethod Physical

# 恢复到新实例
aliyun rds CloneDBInstance \
    --DBInstanceId rm-bp1xxxxxxxx \
    --BackupId 123456789 \
    --DBInstanceClass rds.mysql.s2.large \
    --PayType Postpaid

# 恢复到指定时间
aliyun rds RestoreDBInstance \
    --DBInstanceId rm-bp1xxxxxxxx \
    --BackupId "" \
    --RestoreTime "2024-01-15T12:00:00Z"
```

### 数据库代理（读写分离）

```bash
# 开启数据库代理
aliyun rds CreateDBProxy \
    --DBInstanceId rm-bp1xxxxxxxx \
    --DBProxyInstanceNum 2 \
    --VpcId vpc-bp1xxxxxxxx \
    --VSwitchId vsw-bp1xxxxxxxx

# 配置读写分离权重
aliyun rds ModifyDBProxy \
    --DBInstanceId rm-bp1xxxxxxxx \
    --DBProxyInstanceNum 2 \
    --ReadOnlyInstanceDistributionType Custom \
    --ReadOnlyInstanceWeight.1.DBInstanceId rr-bp1xxxxxxxx \
    --ReadOnlyInstanceWeight.1.Weight 100
```

## 三、注意事项与常见陷阱

1. **高可用版选择**：生产环境至少使用高可用版
2. **白名单配置**：仅允许应用服务器IP访问数据库
3. **连接数限制**：注意最大连接数限制，必要时使用数据库代理
4. **备份策略**：设置自动备份保留策略，手动备份重要节点
5. **性能监控**：关注慢查询和CPU使用率
6. **存储空间**：开启存储自动扩展防止空间不足
7. **大版本升级**：数据库大版本升级需要测试后执行
