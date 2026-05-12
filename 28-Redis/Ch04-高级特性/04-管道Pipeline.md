# 管道Pipeline

## 一、概念说明

Pipeline允许客户端一次性发送多个命令到Redis服务器，减少网络往返次数（RTT），显著提升批量操作性能。

## 二、具体用法

### redis-cli Pipeline

```bash
# 使用--pipe模式
cat commands.txt | redis-cli --pipe

# commands.txt内容（RESP协议格式）
*3\r\n$3\r\nSET\r\n$4\r\nkey1\r\n$6\r\nvalue1\r\n

# 或使用echo
echo -e "SET key1 val1\nSET key2 val2\nGET key1" | redis-cli --pipe
```

### Python示例

```python
import redis

r = redis.Redis(host='localhost', port=6379)

# 使用Pipeline
pipe = r.pipeline()
pipe.set('key1', 'value1')
pipe.set('key2', 'value2')
pipe.get('key1')
pipe.get('key2')

# 执行
results = pipe.execute()
# results: [True, True, b'value1', b'value2']
```

### 性能对比

```bash
# 无Pipeline：每次命令一次RTT
# 10000次SET ≈ 10000次网络往返

# 有Pipeline：批量发送
# 10000次SET ≈ 1次网络往返
# 性能提升10-100倍

# 测试
redis-benchmark -t set -n 100000
# vs
redis-benchmark -t set -n 100000 -P 100
# -P 100表示每次Pipeline包含100个命令
```

## 三、Pipeline vs 事务

```bash
# Pipeline：批量发送，减少RTT
# 命令之间可以插入其他命令

# 事务（MULTI/EXEC）：原子性执行
# 命令之间不能插入其他命令

# Pipeline + 事务：既减少RTT又保证原子性
MULTI
SET key1 val1
SET key2 val2
EXEC
# 所有命令一起发送并原子执行
```

## 四、注意事项

1. **批量大小**：Pipeline不宜过大，建议100-1000个命令
2. **内存占用**：大量命令会占用客户端和服务器内存
3. **无原子性**：Pipeline不保证原子性
4. **返回顺序**：结果按命令发送顺序返回
5. **错误处理**：部分命令失败不影响其他命令执行

## 五、各语言Pipeline示例

### Java (Lettuce)

```java
RedisClient client = RedisClient.create("redis://localhost");
StatefulRedisConnection<String, String> connection = client.connect();
RedisCommands<String, String> commands = connection.sync();

// Pipeline批处理
List<RedisFuture<?>> futures = new ArrayList<>();
connection.setAutoFlushCommands(false);

for (int i = 0; i < 1000; i++) {
    futures.add(asyncCommands.set("key:" + i, "value:" + i));
}

connection.flushCommands();
RedisFutures.awaitAll(futures);
```

### Go (go-redis)

```go
pipe := rdb.Pipeline()

for i := 0; i < 1000; i++ {
    pipe.Set(ctx, fmt.Sprintf("key:%d", i), fmt.Sprintf("value:%d", i), 0)
}

cmds, err := pipe.Exec(ctx)
if err != nil {
    log.Fatal(err)
}

for _, cmd := range cmds {
    fmt.Println(cmd)
}
```

### Node.js (ioredis)

```javascript
const pipeline = redis.pipeline();

// 批量设置
for (let i = 0; i < 1000; i++) {
    pipeline.set(`key:${i}`, `value:${i}`);
}

// 批量获取
for (let i = 0; i < 1000; i++) {
    pipeline.get(`key:${i}`);
}

const results = await pipeline.exec();
// results: [[null, 'OK'], [null, 'OK'], ..., [null, 'value:0'], ...]
```

## 六、Pipeline最佳实践

```bash
# 1. 批量大小选择
# 太小（<10）：网络开销减少不明显
# 太大（>1000）：客户端内存占用增加
# 推荐：100-500个命令

# 2. 内存控制
# Pipeline命令会缓存在客户端
# 1000个SET命令约占用几MB内存
# 需要根据Value大小调整

# 3. 错误处理
results = pipe.execute()
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"命令 {i} 失败: {result}")

# 4. 超时设置
# Pipeline执行时间 = 所有命令中最慢的那个
# 设置合理的socket_timeout

# 5. 监控Pipeline效果
# 对比Pipeline前后的QPS和延迟
redis-benchmark -t set -n 100000
redis-benchmark -t set -n 100000 -P 100
```
