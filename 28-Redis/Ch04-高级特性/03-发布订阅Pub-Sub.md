# 发布订阅Pub/Sub

## 一、概念说明

Redis Pub/Sub是一种消息通信模式，发布者（Publisher）发送消息到频道（Channel），订阅者（Subscriber）接收消息。消息是即时的，不持久化。

## 二、具体用法

### 基本操作

```bash
# 订阅频道
SUBSCRIBE news:sports
# 输出: 等待消息...

# 在另一个终端发布消息
PUBLISH news:sports "比赛开始！"
# 输出: (integer) 1（收到消息的订阅者数）

# 订阅者收到消息
# 1) "message"
# 2) "news:sports"
# 3) "比赛开始！"
```

### 模式订阅

```bash
# 订阅匹配模式
PSUBSCRIBE news:*
# 接收所有news:开头频道的消息

# 发布到任意匹配频道
PUBLISH news:sports "体育新闻"
PUBLISH news:tech "科技新闻"
# 两个消息都会被收到
```

### 取消订阅

```bash
# 取消订阅
UNSUBSCRIBE news:sports

# 取消模式订阅
PUNSUBSCRIBE news:*
```

## 三、实际应用

```bash
# 实时通知
SUBSCRIBE notifications:user:1001
PUBLISH notifications:user:1001 "您有新消息"

# 实时聊天
SUBSCRIBE chat:room1
PUBLISH chat:room1 "大家好"

# 配置更新通知
SUBSCRIBE config:updates
PUBLISH config:updates "reload"
```

## 四、注意事项

1. **消息不持久化**：订阅者离线时的消息会丢失
2. **无消息确认**：发布者不知道订阅者是否收到
3. **阻塞订阅**：SUBSCRIBE会阻塞客户端
4. **性能影响**：大量订阅者会影响发布性能
5. **替代方案**：需要可靠消息使用Stream

## 五、Python Pub/Sub示例

```python
import redis
import threading

r = redis.Redis()

def subscriber():
    """订阅者线程"""
    pubsub = r.pubsub()
    pubsub.subscribe('notifications')

    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"收到消息: {message['data'].decode()}")
        elif message['type'] == 'subscribe':
            print(f"已订阅: {message['channel'].decode()}")

# 启动订阅者
thread = threading.Thread(target=subscriber, daemon=True)
thread.start()

# 发布消息
r.publish('notifications', '系统维护通知')
r.publish('notifications', '服务已恢复')
```

### 模式订阅

```python
pubsub = r.pubsub()

# 订阅多个频道
pubsub.subscribe('channel1', 'channel2')

# 模式订阅
pubsub.psubscribe('news:*')
pubsub.psubscribe('user:*:notifications')

for message in pubsub.listen():
    if message['type'] == 'pmessage':
        print(f"频道: {message['channel'].decode()}")
        print(f"模式: {message['pattern'].decode()}")
        print(f"消息: {message['data'].decode()}")
```

## 六、Pub/Sub vs Stream对比

```bash
# Pub/Sub特点
# 1. 消息即发即弃，不持久化
# 2. 不支持消费者组
# 3. 不支持消息确认
# 4. 适合实时通知

# Stream特点（Redis 5.0+）
# 1. 消息持久化
# 2. 支持消费者组
# 3. 支持消息确认（ACK）
# 4. 支持消息回溯
# 5. 适合可靠消息传递

# 选择建议
# 实时聊天、通知 → Pub/Sub
# 任务队列、日志 → Stream
# 需要消息可靠性 → Stream
# 简单通知场景 → Pub/Sub
```

## 七、Node.js Pub/Sub

```javascript
const Redis = require('ioredis');

const subscriber = new Redis();
const publisher = new Redis();

// 订阅
subscriber.subscribe('notifications', (err, count) => {
    if (err) console.error(err);
    console.log(`已订阅 ${count} 个频道`);
});

// 接收消息
subscriber.on('message', (channel, message) => {
    console.log(`频道 ${channel}: ${message}`);
});

// 模式订阅
subscriber.psubscribe('news:*');
subscriber.on('pmessage', (pattern, channel, message) => {
    console.log(`模式 ${pattern}, 频道 ${channel}: ${message}`);
});

// 发布
publisher.publish('notifications', 'Hello World');
```

## 八、生产环境注意事项

```bash
# 1. 使用Redis Stream替代Pub/Sub处理关键消息
# 2. Pub/Sub适合实时性要求高但允许丢失的场景
# 3. 监控订阅者数量，防止消息无人消费
# 4. 使用TLS加密敏感消息
# 5. 在Sentinel/Cluster模式下，Pub/Sub可正常工作
```
