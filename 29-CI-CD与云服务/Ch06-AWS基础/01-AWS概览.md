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

## 五、AWS核心服务详解

### EC2实例管理

```bash
# 启动EC2实例
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.micro \
  --key-name my-key-pair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=web-server}]'

# 管理实例
aws ec2 describe-instances
aws ec2 start-instances --instance-ids i-1234567890abcdef0
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

### S3存储操作

```bash
# 创建Bucket
aws s3 mb s3://my-unique-bucket-name

# 上传文件
aws s3 cp ./dist s3://my-bucket/ --recursive
aws s3 sync ./dist s3://my-bucket/

# 设置静态网站托管
aws s3 website s3://my-bucket/ --index-document index.html --error-document error.html

# 配置Bucket策略
aws s3api put-bucket-policy --bucket my-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicRead",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}'
```

### ECS/Fargate容器部署

```bash
# 创建ECS集群
aws ecs create-cluster --cluster-name my-cluster

# 注册任务定义
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 创建服务
aws ecs create-service \
  --cluster my-cluster \
  --service-name my-service \
  --task-definition my-task \
  --desired-count 3 \
  --launch-type FARGATE

# 更新服务
aws ecs update-service \
  --cluster my-cluster \
  --service my-service \
  --force-new-deployment
```

```json
// task-definition.json
{
  "family": "my-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [{
    "name": "myapp",
    "image": "123456.dkr.ecr.us-east-1.amazonaws.com/myapp:latest",
    "portMappings": [{
      "containerPort": 3000,
      "protocol": "tcp"
    }],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/myapp",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

### Lambda无服务器函数

```bash
# 创建Lambda函数
aws lambda create-function \
  --function-name my-function \
  --runtime nodejs20.x \
  --handler index.handler \
  --role arn:aws:iam::123456789:role/lambda-role \
  --zip-file fileb://function.zip

# 更新函数代码
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip

# 调用函数
aws lambda invoke \
  --function-name my-function \
  --payload '{"key": "value"}' \
  output.json
```

## 六、IAM权限管理

```json
// 最小权限策略示例
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
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices"
      ],
      "Resource": "arn:aws:ecs:us-east-1:123456789:service/my-cluster/*"
    }
  ]
}
```

```bash
# 创建IAM角色
aws iam create-role --role-name my-role --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name my-role --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# OIDC身份联合（用于CI/CD）
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

## 七、CloudFormation基础设施即代码

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Web Application Stack

Parameters:
  Environment:
    Type: String
    Default: staging
    AllowedValues: [staging, production]

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-vpc"

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub "${Environment}-cluster"

  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub "${Environment}-alb"
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

Outputs:
  LoadBalancerDNS:
    Value: !GetAtt LoadBalancer.DNSName
    Description: Load Balancer DNS Name
```

```bash
# 部署CloudFormation栈
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name my-stack \
  --parameter-overrides Environment=production \
  --capabilities CAPABILITY_IAM

# 查看栈输出
aws cloudformation describe-stacks --stack-name my-stack --query 'Stacks[0].Outputs'
```

## 八、AWS成本优化

```bash
# 查看成本和使用情况
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE

# 设置预算告警
aws budgets create-budget \
  --account-id 123456789 \
  --budget '{
    "BudgetName": "monthly-budget",
    "BudgetLimit": { "Amount": "100", "Unit": "USD" },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{
      "SubscriptionType": "EMAIL",
      "Address": "admin@example.com"
    }]
  }]'
```

## 九、AWS CLI配置管理

```bash
# 多Profile配置
aws configure --profile dev
aws configure --profile staging
aws configure --profile production

# 使用特定Profile
aws s3 ls --profile production
AWS_PROFILE=production aws ec2 describe-instances

# 配置文件位置
# ~/.aws/config - 配置
# ~/.aws/credentials - 凭据

# 环境变量配置
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1
```

## 十、AWS CI/CD服务

```bash
# AWS CodePipeline
aws codepipeline create-pipeline --cli-input-json file://pipeline.json

# AWS CodeBuild
aws codebuild create-project --name my-build \
  --source type=GITHUB,location=https://github.com/org/repo.git \
  --artifacts type=S3,location=my-bucket \
  --environment type=LINUX_CONTAINER,computeType=BUILD_GENERAL1_SMALL,image=aws/codebuild/standard:5.0

# AWS CodeDeploy
aws deploy create-application --application-name my-app
aws deploy create-deployment-group \
  --application-name my-app \
  --deployment-group-name my-dg \
  --deployment-config-name CodeDeployDefault.OneAtATime \
  --ec2-tag-filters Key=Name,Value=web-server,Type=KEY_AND_VALUE
```
