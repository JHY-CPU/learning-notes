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
4. **Promise支持**：ioredis原生支持Promise
