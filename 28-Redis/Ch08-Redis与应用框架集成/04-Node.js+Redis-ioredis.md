# Node.js + Redis (ioredis)

## 一、概念说明

ioredis是Node.js最流行的Redis客户端，支持集群、哨兵、Pipeline等功能。

## 二、基本使用

```javascript
const Redis = require('ioredis');

// 单机连接
const redis = new Redis({
    host: '192.168.1.100',
    port: 6379,
    password: 'yourpassword',
    db: 0,
    retryStrategy(times) {
        return Math.min(times * 50, 2000);
    }
});

// 基本操作
await redis.set('key', 'value');
await redis.get('key');
await redis.del('key');

// Hash操作
await redis.hset('user:1', 'name', '张三');
await redis.hget('user:1', 'name');
await redis.hgetall('user:1');

// List操作
await redis.lpush('queue', 'task1');
await redis.rpop('queue');
await redis.lrange('queue', 0, -1);
```

## 三、集群连接

```javascript
const cluster = new Redis.Cluster([
    { host: '192.168.1.100', port: 7000 },
    { host: '192.168.1.101', port: 7001 },
    { host: '192.168.1.102', port: 7002 }
], {
    redisOptions: { password: 'yourpassword' }
});

await cluster.set('key', 'value');
await cluster.get('key');
```

## 四、Pipeline与事务

```javascript
// Pipeline
const pipeline = redis.pipeline();
for (let i = 0; i < 1000; i++) {
    pipeline.set(`key:${i}`, `value:${i}`);
}
await pipeline.exec();

// 事务
const multi = redis.multi();
multi.set('key1', 'value1');
multi.set('key2', 'value2');
await multi.exec();
```

## 五、注意事项

1. **连接事件**：监听connect/error/close事件
2. **重试策略**：合理设置重试策略
3. **内存泄漏**：不用时调用redis.quit()
## 六、Pub/Sub使用

```javascript
const Redis = require('ioredis');

// 订阅者
const subscriber = new Redis();
const publisher = new Redis();

// 订阅频道
subscriber.subscribe('notifications', (err, count) => {
    if (err) console.error(err);
    console.log(`已订阅 ${count} 个频道`);
});

// 接收消息
subscriber.on('message', (channel, message) => {
    console.log(`频道 ${channel}: ${message}`);
});

// 模式订阅
subscriber.psubscribe('user:*', (err) => {
    if (err) console.error(err);
});

subscriber.on('pmessage', (pattern, channel, message) => {
    console.log(`模式 ${pattern}, 频道 ${channel}: ${message}`);
});

// 发布消息
publisher.publish('notifications', JSON.stringify({
    type: 'info',
    message: '系统维护通知'
}));
```

## 七、Stream消费者组

```javascript
async function streamConsumer() {
    const redis = new Redis();
    
    // 创建消费者组
    try {
        await redis.xgroup('CREATE', 'orders', 'order-group', '0', 'MKSTREAM');
    } catch (e) {
        // 组已存在
    }
    
    // 消费消息
    while (true) {
        const results = await redis.xreadgroup(
            'GROUP', 'order-group', 'consumer1',
            'COUNT', 10,
            'BLOCK', 5000,
            'STREAMS', 'orders', '>'
        );
        
        if (results) {
            for (const [stream, messages] of results) {
                for (const [id, fields] of messages) {
                    try {
                        // 处理消息
                        console.log(`处理消息: ${id}`);
                        
                        // 确认消息
                        await redis.xack('orders', 'order-group', id);
                    } catch (e) {
                        console.error(`处理失败: ${e}`);
                    }
                }
            }
        }
    }
}

streamConsumer();
```

## 八、错误处理与重连

```javascript
const Redis = require('ioredis');

const redis = new Redis({
    host: '192.168.1.100',
    port: 6379,
    password: 'yourpassword',
    retryStrategy(times) {
        const delay = Math.min(times * 100, 3000);
        console.log(`重连尝试 ${times}, 延迟 ${delay}ms`);
        return delay;
    },
    maxRetriesPerRequest: 3,
    enableReadyCheck: true,
    reconnectOnError(err) {
        const targetErrors = ['READONLY', 'LOADING'];
        return targetErrors.some(e => err.message.includes(e));
    }
});

// 事件监听
redis.on('connect', () => console.log('已连接'));
redis.on('ready', () => console.log('准备就绪'));
redis.on('error', (err) => console.error('错误:', err));
redis.on('close', () => console.log('连接关闭'));
redis.on('reconnecting', (delay) => console.log(`重连中, 延迟: ${delay}ms`));

// 优雅退出
process.on('SIGINT', async () => {
    await redis.quit();
    process.exit(0);
});
```
