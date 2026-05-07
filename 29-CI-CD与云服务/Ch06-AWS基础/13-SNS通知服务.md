# SNS通知服务

## 一、概念说明

SNS（Simple Notification Service）是AWS的发布/订阅消息服务，支持将消息推送到多个订阅者。适用于解耦和扇出消息场景。

| 概念 | 说明 |
|------|------|
| Topic | 消息主题，消息的逻辑访问点 |
| Publisher | 消息发布者 |
| Subscriber | 消息订阅者 |
| Subscription | 订阅关系 |

| 协议 | 说明 |
|------|------|
| HTTP/HTTPS | Webhook推送 |
| Email | 邮件通知 |
| SMS | 短信通知 |
| SQS | 队列订阅 |
| Lambda | 函数触发 |
| Application | 移动推送 |

## 二、具体用法

### 创建主题和订阅

```bash
# 创建主题
aws sns create-topic --name my-notifications

# 创建FIFO主题
aws sns create-topic \
    --name my-ordered.fifo \
    --attributes '{"FifoTopic":"true","ContentBasedDeduplication":"true"}'

# 订阅Email
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --protocol email \
    --notification-endpoint admin@example.com

# 订阅SQS队列
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --protocol sqs \
    --notification-endpoint arn:aws:sqs:us-east-1:123456789012:my-queue

# 订阅Lambda函数
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --protocol lambda \
    --notification-endpoint arn:aws:lambda:us-east-1:123456789012:function:my-func
```

### 发布消息

```bash
# 简单消息
aws sns publish \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --subject "部署通知" \
    --message "应用v2.0已成功部署到生产环境"

# 按协议不同内容
aws sns publish \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --message '{"default":"默认消息","email":"邮件内容","sms":"短信内容"}' \
    --message-structure json

# 带消息属性（用于过滤）
aws sns publish \
    --topic-arn arn:aws:sns:us-east-1:123456789012:my-notifications \
    --message "告警消息" \
    --message-attributes '{
        "severity": {
            "DataType": "String",
            "StringValue": "high"
        }
    }'
```

### 消息过滤策略

```bash
# 设置订阅过滤策略
aws sns set-subscription-attributes \
    --subscription-arn arn:aws:sns:us-east-1:123456789012:my-notifications:abc123 \
    --attribute-name FilterPolicy \
    --attribute-value '{"severity": ["high", "critical"]}'
```

### CloudWatch告警集成

```bash
# CloudWatch告警发送到SNS
aws cloudwatch put-metric-alarm \
    --alarm-name HighCPU \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:my-notifications
```

## 三、注意事项与常见陷阱

1. **消息持久性**：SNS不持久化消息，需配合SQS保证投递
2. **订阅确认**：Email/HTTP订阅需要确认
3. **扇出限制**：每个主题最多1000万订阅者
4. **消息过滤**：使用过滤策略减少不必要的消息投递
5. **死信队列**：配置SQS死信队列处理投递失败
6. **SMS成本**：SMS通知按条计费，注意成本控制
7. **主题策略**：配置主题策略控制谁能发布消息
