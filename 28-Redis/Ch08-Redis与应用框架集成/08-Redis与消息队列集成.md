# Redis与消息队列集成

## 一、概念说明

Redis可以作为轻量级消息队列，与各种框架集成实现任务队列、事件处理等。

## 二、Bull (Node.js)

```javascript
const Queue = require('bull');

const emailQueue = new Queue('email', {
    redis: { host: '192.168.1.100', port: 6379 }
});

// 添加任务
await emailQueue.add({
    to: 'user@example.com',
    subject: 'Hello',
    body: 'World'
}, {
    delay: 5000,  // 延迟5秒
    attempts: 3   // 重试3次
});

// 处理任务
emailQueue.process(async (job) => {
    const { to, subject, body } = job.data;
    await sendEmail(to, subject, body);
});
```

## 三、Celery (Python)

```python
from celery import Celery

app = Celery('tasks', broker='redis://192.168.1.100:6379/0')

@app.task
def send_email(to, subject, body):
    # 发送邮件
    pass

# 调用任务
send_email.delay('user@example.com', 'Hello', 'World')
```

## 四、Spring Boot + Redisson

```java
RQueue<Task> queue = redisson.getQueue("task:queue");
queue.offer(new Task("process", data));

RBlockingQueue<Task> blockingQueue = redisson.getBlockingQueue("task:queue");
Task task = blockingQueue.take();  // 阻塞等待
```

## 五、注意事项

1. **消息持久化**：Stream比List更适合持久化
2. **消费者组**：Stream支持消费者组
3. **消息确认**：确保消息不丢失
4. **死信队列**：处理失败的消息
## 六、Redis Streams作为消息队列

```python
import redis
import json
import time

r = redis.Redis()

class StreamMessageQueue:
    """基于Redis Stream的消息队列"""
    
    def __init__(self, stream_name, group_name, consumer_name):
        self.stream = stream_name
        self.group = group_name
        self.consumer = consumer_name
        
        # 创建消费者组
        try:
            r.xgroup_create(self.stream, self.group, id='0', mkstream=True)
        except redis.ResponseError:
            pass
    
    def publish(self, message, max_len=10000):
        """发布消息"""
        return r.xadd(self.stream, {'data': json.dumps(message)}, maxlen=max_len)
    
    def consume(self, count=10, block=5000):
        """消费消息"""
        messages = r.xreadgroup(
            self.group, self.consumer,
            {self.stream: '>'},
            count=count, block=block
        )
        
        results = []
        for stream, msgs in messages:
            for msg_id, data in msgs:
                results.append({
                    'id': msg_id,
                    'data': json.loads(data[b'data'])
                })
        return results
    
    def ack(self, msg_id):
        """确认消息"""
        return r.xack(self.stream, self.group, msg_id)
    
    def pending(self):
        """查看待处理消息"""
        return r.xpending(self.stream, self.group)
    
    def process_with_ack(self, handler, count=10):
        """处理消息并自动确认"""
        messages = self.consume(count=count)
        
        for msg in messages:
            try:
                handler(msg['data'])
                self.ack(msg['id'])
            except Exception as e:
                print(f"处理失败: {e}")

# 使用
queue = StreamMessageQueue('orders', 'order_group', 'consumer1')

# 发布
queue.publish({'order_id': 1001, 'amount': 99.9})

# 消费
def handle_order(data):
    print(f"处理订单: {data}")

queue.process_with_ack(handle_order)
```

## 七、死信队列处理

```python
class DeadLetterHandler:
    """死信队列处理器"""
    
    def __init__(self, main_stream, dlq_stream, max_retries=3):
        self.main_stream = main_stream
        self.dlq_stream = dlq_stream
        self.max_retries = max_retries
        self.r = redis.Redis()
    
    def process_with_retry(self, msg_id, data, handler):
        """带重试的处理"""
        for attempt in range(self.max_retries):
            try:
                handler(data)
                return True
            except Exception as e:
                print(f"重试 {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)
        
        # 进入死信队列
        self.r.xadd(self.dlq_stream, {
            'original_id': msg_id,
            'data': json.dumps(data),
            'failed_at': time.time()
        })
        return False
```

## 八、队列监控

```python
def monitor_queue(stream_name):
    """监控队列状态"""
    info = r.xinfo_stream(stream_name)
    
    print(f"队列: {stream_name}")
    print(f"  长度: {info['length']}")
    print(f"  第一条: {info['first-entry'][0] if info['first-entry'] else 'N/A'}")
    print(f"  最后一条: {info['last-entry'][0] if info['last-entry'] else 'N/A'}")
    
    # 消费者组信息
    groups = r.xinfo_groups(stream_name)
    for group in groups:
        print(f"  组: {group['name'].decode()}")
        print(f"    消费者: {group['consumers']}")
        print(f"    待处理: {group['pending']}")
```
