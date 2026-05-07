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
5. **监控队列积压**：防止内存溢出
