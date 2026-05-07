# AWS最佳实践

## 一、概念说明

AWS最佳实践涵盖了架构设计、安全、成本优化、运维等方面的推荐做法。遵循Well-Architected Framework五大支柱可构建可靠的云系统。

| 支柱 | 核心目标 |
|------|----------|
| 运维卓越 | 自动化和持续改进 |
| 安全 | 保护数据和系统 |
| 可靠性 | 从故障中恢复 |
| 性能效率 | 高效使用资源 |
| 成本优化 | 避免不必要的支出 |

## 二、具体用法

### 高可用架构

```yaml
# 多AZ部署架构
Architecture:
  Load_Balancer:
    - ALB跨多个可用区
    - 健康检查自动剔除不健康实例
  Application:
    - Auto Scaling Group跨3个AZ
    - 最小实例数2，最大实例数10
  Database:
    - RDS多AZ部署
    - 读写分离（读副本）
  Cache:
    - ElastiCache Redis集群模式
    - 跨AZ副本
```

### 安全加固

```bash
# 1. 启用GuardDuty（威胁检测）
aws guardduty create-detector --enable

# 2. 启用Security Hub
aws securityhub enable-security-hub

# 3. 配置AWS Config（合规检查）
aws configservice put-configuration-recorder \
    --configuration-recorder name=default,roleArn=arn:aws:iam::role/aws-service-role

# 4. 启用CloudTrail（审计日志）
aws cloudtrail create-trail \
    --name my-trail \
    --s3-bucket-name my-audit-bucket \
    --is-multi-region-trail \
    --enable-log-file-validation
```

### 成本优化策略

```bash
# 成本分配标签
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-02-01 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=TAG,Key=Environment

# 预留实例建议
aws ce get-reservation-purchase-recommendation \
    --service EC2 \
    --account-scope PAYER
```

### 自动化运维

```python
# 使用SSM自动化运维
import boto3

ssm = boto3.client('ssm')

# 执行自动化文档
response = ssm.start_automation_execution(
    DocumentName='AWS-UpdateLinuxAmi',
    Parameters={
        'InstanceId': ['i-1234567890abcdef0'],
        'AutomationAssumeRole': ['arn:aws:iam::role/SSMRole']
    }
)

# 批量执行命令
response = ssm.send_command(
    InstanceIds=['i-123456', 'i-789012'],
    DocumentName='AWS-RunShellScript',
    Parameters={'commands': ['yum update -y']}
)
```

### 备份策略

```json
{
    "BackupPlan": {
        "BackupPlanName": "DailyBackup",
        "Rules": [{
            "RuleName": "DailyBackups",
            "TargetBackupVaultName": "Default",
            "ScheduleExpression": "cron(0 5 ? * * *)",
            "Lifecycle": {
                "DeleteAfterDays": 90,
                "MoveToColdStorageAfterDays": 30
            },
            "CopyActions": [{
                "DestinationBackupVaultArn": "arn:aws:backup:us-west-2:123456789012:backup-vault/DR",
                "Lifecycle": {"DeleteAfterDays": 365}
            }]
        }]
    }
}
```

## 三、注意事项与常见陷阱

1. **多区域部署**：关键业务考虑跨区域灾备
2. **基础设施即代码**：所有资源使用CloudFormation/Terraform管理
3. **密钥管理**：使用Secrets Manager或Parameter Store管理密钥
4. **日志集中管理**：使用CloudWatch Logs或ELK集中收集日志
5. **资源标记**：所有资源添加成本归属标签
6. **变更管理**：使用Config追踪资源配置变更
7. **定期审查**：每月审查安全、成本和架构
