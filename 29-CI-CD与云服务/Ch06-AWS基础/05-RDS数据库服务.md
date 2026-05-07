# RDS数据库服务

## 一、概念说明

RDS（Relational Database Service）是AWS的托管关系型数据库服务，支持多种数据库引擎，自动处理备份、补丁、故障恢复等运维任务。

| 引擎 | 版本 | 适用场景 |
|------|------|----------|
| MySQL | 5.7, 8.0 | Web应用 |
| PostgreSQL | 12-15 | 地理空间/JSON |
| MariaDB | 10.6+ | MySQL替代 |
| Oracle | 19c | 企业级应用 |
| SQL Server | 2019 | .NET应用 |

## 二、具体用法

### 创建数据库实例

```bash
# 创建RDS实例
aws rds create-db-instance \
    --db-instance-identifier mydb \
    --db-instance-class db.t3.micro \
    --engine mysql \
    --master-username admin \
    --master-user-password mypassword123 \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-12345678 \
    --db-subnet-group-name my-subnet-group \
    --backup-retention-period 7 \
    --multi-az \
    --storage-encrypted
```

### 连接数据库

```bash
# 获取连接终端节点
aws rds describe-db-instances \
    --db-instance-identifier mydb \
    --query 'DBInstances[0].Endpoint.Address'

# 连接MySQL
mysql -h mydb.xxxx.us-east-1.rds.amazonaws.com -u admin -p

# 连接PostgreSQL
psql -h mydb.xxxx.us-east-1.rds.amazonaws.com -U admin -d mydb
```

### 备份和恢复

```bash
# 创建手动快照
aws rds create-db-snapshot \
    --db-instance-identifier mydb \
    --db-snapshot-identifier mydb-snapshot-202401

# 从快照恢复
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier mydb-restored \
    --db-snapshot-identifier mydb-snapshot-202401 \
    --db-instance-class db.t3.micro

# 时间点恢复
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier mydb \
    --target-db-instance-identifier mydb-pitr \
    --restore-time 2024-01-15T12:00:00Z
```

### 只读副本和监控

```bash
# 创建只读副本
aws rds create-db-instance-read-replica \
    --db-instance-identifier mydb-replica \
    --source-db-instance-identifier mydb \
    --db-instance-class db.t3.micro

# 启用增强监控
aws rds modify-db-instance \
    --db-instance-identifier mydb \
    --monitoring-interval 60 \
    --monitoring-role-arn arn:aws:iam::123456789012:role/rds-monitoring
```

## 三、注意事项与常见陷阱

1. **多可用区部署**：生产环境必须启用多AZ保证高可用
2. **存储自动扩展**：启用存储自动扩展避免空间不足
3. **备份策略**：合理设置备份保留期，手动快照保留长期备份
4. **安全组规则**：仅允许应用服务器IP访问数据库端口
5. **参数组优化**：根据工作负载调整数据库参数
6. **连接池管理**：推荐使用RDS Proxy管理连接池
7. **成本控制**：开发环境使用较小实例，避免不必要的多AZ
