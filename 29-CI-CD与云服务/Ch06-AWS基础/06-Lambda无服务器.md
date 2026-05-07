# Lambda无服务器

## 一、概念说明

AWS Lambda是无服务器计算服务，无需管理服务器即可运行代码。按执行时间计费，支持事件驱动架构。

| 概念 | 限制 | 说明 |
|------|------|------|
| 内存 | 128MB-10240MB | 配置越大CPU越高 |
| 超时 | 最大15分钟 | 执行时间限制 |
| 包大小 | 50MB压缩/250MB解压 | 部署包限制 |
| 并发 | 默认1000 | 可申请提高 |
| 环境变量 | 4KB | 配置限制 |

## 二、具体用法

### 创建Lambda函数

```python
# lambda_function.py
import json
import boto3

def lambda_handler(event, context):
    """Lambda入口函数"""
    # 解析事件
    body = json.loads(event.get('body', '{}'))
    name = body.get('name', 'World')

    # 业务逻辑
    message = f"Hello, {name}!"

    # 返回响应
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'message': message})
    }
```

```bash
# 打包部署
zip function.zip lambda_function.py

# 创建函数
aws lambda create-function \
    --function-name my-function \
    --runtime python3.11 \
    --role arn:aws:iam::123456789012:role/lambda-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://function.zip \
    --timeout 30 \
    --memory-size 256

# 更新函数代码
aws lambda update-function-code \
    --function-name my-function \
    --zip-file fileb://function.zip
```

### 配置触发器

```bash
# API Gateway触发器
aws lambda add-permission \
    --function-name my-function \
    --statement-id apigateway \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com

# S3触发器
aws s3api put-bucket-notification-configuration \
    --bucket my-bucket \
    --notification-configuration '{
        "LambdaFunctionConfigurations": [{
            "LambdaFunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:my-function",
            "Events": ["s3:ObjectCreated:*"]
        }]
    }'

# CloudWatch定时触发
aws events put-rule \
    --name "every-5-minutes" \
    --schedule-expression "rate(5 minutes)"
```

### 层（Layer）管理

```bash
# 创建层
zip -r layer.zip python/
aws lambda publish-layer-version \
    --layer-name my-dependencies \
    --zip-file fileb://layer.zip \
    --compatible-runtimes python3.11

# 附加层到函数
aws lambda update-function-configuration \
    --function-name my-function \
    --layers arn:aws:lambda:us-east-1:123456789012:layer:my-dependencies:1
```

## 三、注意事项与常见陷阱

1. **冷启动问题**：首次调用或长时间未调用会产生延迟
2. **内存配置**：内存影响CPU分配，需平衡性能和成本
3. **VPC配置**：访问VPC资源需配置子网和安全组
4. **超时设置**：合理设置超时时间，避免长时间运行
5. **并发限制**：注意并发数限制，必要时申请提高配额
6. **依赖打包**：大型依赖考虑使用Lambda层或容器镜像
7. **环境变量**：敏感信息使用环境变量加密存储
