# SQS消息队列

## 一、概念说明

SQS（Simple Queue Service）是AWS的托管消息队列服务，支持分布式系统解耦和异步消息处理。提供标准队列和FIFO队列两种类型。

| 队列类型 | 特点 | 适用场景 |
|----------|------|----------|
| Standard | 最大吞吐量，可能重复 | 高吞吐、可容忍重复 |
| FIFO | 严格顺序，精确一次 | 事务性、顺序敏感 |

## 二、具体用法

### 创建和操作队列

```bash
# 创建标准队列
aws sqs create-queue \
    --queue-name my-queue \
    --attributes '{
        "VisibilityTimeout": "300",
        "MessageRetentionPeriod": "345600",
        "ReceiveMessageWaitTimeSeconds": "20"
    }'

# 创建FIFO队列
aws sqs create-queue \
    --queue-name my-queue.fifo \
    --attributes '{"FifoQueue": "true", "ContentBasedDeduplication": "true"}'
```

### 发送和接收消息

```bash
# 发送消息
aws sqs send-message \
    --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue \
    --message-body '{"orderId": "12345", "action": "process"}' \
    --message-attributes '{
        "Priority": {
            "DataType": "String",
            "StringValue": "high"
        }
    }'

# 批量发送
aws sqs send-message-batch \
    --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue \
    --entries '[
        {"Id": "1", "MessageBody": "message1"},
        {"Id": "2", "MessageBody": "message2"},
        {"Id": "3", "MessageBody": "message3"}
    ]'

# 接收消息
aws sqs receive-message \
    --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue \
    --max-number-of-messages 10 \
    --wait-time-seconds 20
```

### 消费者处理

```python
import boto3
import json

sqs = boto3.client('sqs')
queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789012/my-queue'

def process_messages():
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20
        )

        messages = response.get('Messages', [])
        for message in messages:
            try:
                body = json.loads(message['Body'])
                # 处理业务逻辑
                handle_order(body)
                # 处理成功后删除消息
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
            except Exception as e:
                print(f"处理失败: {e}")
                # 消息将重新可见
```

### 死信队列

```bash
# 创建死信队列
aws sqs create-queue --queue-name my-queue-dlq

# 配置主队列的死信队列
aws sqs set-queue-attributes \
    --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/my-queue \
    --attributes '{
        "RedrivePolicy": "{\"deadLetterTargetArn\":\"arn:aws:sqs:us-east-1:123456789012:my-queue-dlq\",\"maxReceiveCount\":3}"
    }'
```

## 三、注意事项与常见陷阱

1. **消息重复**：标准队列可能重复，业务逻辑需要幂等
2. **可见性超时**：设置合理的超时，避免消息被重复处理
3. **长轮询**：使用长轮询（WaitTimeSeconds）减少空轮询成本
4. **消息大小**：单条消息最大256KB，大消息使用S3存储引用
5. **死信队列**：始终配置死信队列处理失败消息
6. **批量操作**：使用批量API减少请求次数和成本
7. **FIFO限制**：FIFO队列有300消息/秒的吞吐限制
