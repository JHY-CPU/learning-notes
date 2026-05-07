# IAM权限管理

## 一、概念说明

IAM（Identity and Access Management）是AWS的身份与访问管理服务，用于控制谁可以访问AWS资源以及如何访问。核心概念包括用户、组、角色和策略。

| 概念 | 说明 | 用途 |
|------|------|------|
| User | 具体的AWS用户 | 个人账号访问 |
| Group | 用户的逻辑分组 | 批量权限管理 |
| Role | 可被假设的身份 | 跨账号/服务访问 |
| Policy | 权限定义文档 | 细粒度权限控制 |

## 二、具体用法

### 创建用户和组

```bash
# 创建IAM组
aws iam create-group --group-name Developers

# 创建IAM用户
aws iam create-user --user-name zhangsan

# 将用户添加到组
aws iam add-user-to-group --user-name zhangsan --group-name Developers

# 创建访问密钥
aws iam create-access-key --user-name zhangsan
```

### 策略文档

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ]
        }
    ]
}
```

### 创建和附加策略

```bash
# 创建自定义策略
aws iam create-policy \
    --policy-name S3AccessPolicy \
    --policy-document file://policy.json

# 将策略附加到组
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::123456789012:policy/S3AccessPolicy
```

### IAM角色

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

```bash
# 创建角色
aws iam create-role \
    --role-name EC2S3AccessRole \
    --assume-role-policy-document file://trust-policy.json

# 创建实例配置文件
aws iam create-instance-profile \
    --instance-profile-name EC2Profile

aws iam add-role-to-instance-profile \
    --instance-profile-name EC2Profile \
    --role-name EC2S3AccessRole
```

## 三、注意事项与常见陷阱

1. **最小权限原则**：只授予完成任务所需的最小权限
2. **避免使用根密钥**：创建管理员用户代替根账号操作
3. **定期轮换密钥**：建议每90天轮换一次Access Key
4. **启用MFA**：为所有用户启用多因素认证
5. **使用角色代替密钥**：EC2/Lambda使用IAM角色而非硬编码密钥
6. **策略版本管理**：使用策略版本追踪变更历史
7. **审计日志**：启用CloudTrail记录所有IAM操作
