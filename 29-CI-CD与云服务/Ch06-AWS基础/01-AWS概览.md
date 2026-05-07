# AWS概览

## 一、核心服务

```bash
# 计算
EC2 - 虚拟服务器
Lambda - 无服务器函数
ECS/EKS - 容器编排

# 存储
S3 - 对象存储
EBS - 块存储
EFS - 文件存储

# 数据库
RDS - 关系型数据库
DynamoDB - NoSQL数据库
ElastiCache - 缓存

# 网络
VPC - 虚拟网络
CloudFront - CDN
Route 53 - DNS

# 安全
IAM - 身份管理
KMS - 密钥管理
WAF - Web防火墙
```

## 二、免费层

```bash
# EC2 - 750小时/月 t2.micro
# S3 - 5GB存储
# RDS - 750小时/月 db.t2.micro
# Lambda - 100万次请求/月
# DynamoDB - 25GB存储
```

## 三、基本操作

```bash
# 安装AWS CLI
pip install awscli

# 配置
aws configure
# AWS Access Key ID
# AWS Secret Access Key
# Default region
# Default output format

# 基本命令
aws s3 ls
aws ec2 describe-instances
aws lambda list-functions
```

## 四、注意事项

1. **安全密钥**：保护Access Key
2. **区域选择**：选择靠近用户的区域
3. **成本控制**：设置账单告警
4. **资源标签**：使用标签管理资源
