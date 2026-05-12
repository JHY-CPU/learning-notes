# Go + go-redis

## 一、概念说明

go-redis是Go语言最流行的Redis客户端，支持集群、哨兵、Pipeline等。

## 二、基本使用

```go
package main

import (
    "context"
    "github.com/go-redis/redis/v9"
    "time"
)

func main() {
    ctx := context.Background()
    
    rdb := redis.NewClient(&redis.Options{
        Addr:     "192.168.1.100:6379",
        Password: "yourpassword",
        DB:       0,
        PoolSize: 20,
    })
    
    // 基本操作
    rdb.Set(ctx, "key", "value", time.Hour)
    val, _ := rdb.Get(ctx, "key").Result()
    
    // Hash操作
    rdb.HSet(ctx, "user:1", "name", "张三")
    name, _ := rdb.HGet(ctx, "user:1", "name").Result()
    
    // List操作
    rdb.LPush(ctx, "queue", "task1")
    task, _ := rdb.RPop(ctx, "queue").Result()
}
```

## 三、集群连接

```go
rdb := redis.NewClusterClient(&redis.ClusterOptions{
    Addrs: []string{
        "192.168.1.100:7000",
        "192.168.1.101:7001",
        "192.168.1.102:7002",
    },
    Password: "yourpassword",
    PoolSize: 20,
})
```

## 四、Pipeline

```go
pipe := rdb.Pipeline()
for i := 0; i < 1000; i++ {
    pipe.Set(ctx, fmt.Sprintf("key:%d", i), fmt.Sprintf("value:%d", i), 0)
}
_, err := pipe.Exec(ctx)
```

## 五、注意事项

1. **Context**：使用context控制超时
2. **连接池**：合理设置PoolSize
3. **错误处理**：检查每个操作的错误
## 六、Stream消费者组

```go
package main

import (
    "context"
    "fmt"
    "github.com/go-redis/redis/v9"
)

func consumeStream(ctx context.Context, rdb *redis.Client) {
    // 创建消费者组
    rdb.XGroupCreateMkStream(ctx, "orders", "order-group", "0")
    
    for {
        // 消费消息
        results, err := rdb.XReadGroup(ctx, &redis.XReadGroupArgs{
            Group:    "order-group",
            Consumer: "consumer1",
            Streams:  []string{"orders", ">"},
            Count:    10,
            Block:    5 * time.Second,
        }).Result()
        
        if err == redis.Nil {
            continue
        }
        if err != nil {
            fmt.Printf("读取错误: %v\n", err)
            continue
        }
        
        for _, stream := range results {
            for _, msg := range stream.Messages {
                fmt.Printf("处理消息: %v\n", msg.Values)
                
                // 确认消息
                rdb.XAck(ctx, "orders", "order-group", msg.ID)
            }
        }
    }
}
```

## 七、分布式锁

```go
package main

import (
    "context"
    "github.com/go-redis/redis/v9"
    "github.com/google/uuid"
    "time"
)

type RedisLock struct {
    rdb    *redis.Client
    key    string
    token  string
    expire time.Duration
}

func NewRedisLock(rdb *redis.Client, key string, expire time.Duration) *RedisLock {
    return &RedisLock{
        rdb:    rdb,
        key:    "lock:" + key,
        token:  uuid.New().String(),
        expire: expire,
    }
}

func (l *RedisLock) Acquire(ctx context.Context, timeout time.Duration) bool {
    end := time.Now().Add(timeout)
    for time.Now().Before(end) {
        ok, err := l.rdb.SetNX(ctx, l.key, l.token, l.expire).Result()
        if err == nil && ok {
            return true
        }
        time.Sleep(100 * time.Millisecond)
    }
    return false
}

func (l *RedisLock) Release(ctx context.Context) bool {
    lua := `
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    `
    result, _ := l.rdb.Eval(ctx, lua, []string{l.key}, l.token).Int()
    return result == 1
}

// 使用
func main() {
    ctx := context.Background()
    rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
    
    lock := NewRedisLock(rdb, "resource:1001", 30*time.Second)
    if lock.Acquire(ctx, 10*time.Second) {
        defer lock.Release(ctx)
        fmt.Println("获取锁成功")
    }
}
```

## 八、缓存模式

```go
package main

import (
    "context"
    "encoding/json"
    "github.com/go-redis/redis/v9"
    "time"
)

type CacheAside struct {
    rdb *redis.Client
}

func (c *CacheAside) Get(ctx context.Context, key string, queryFunc func() (interface{}, error), ttl time.Duration) (interface{}, error) {
    // 查缓存
    val, err := c.rdb.Get(ctx, key).Result()
    if err == nil {
        var result interface{}
        json.Unmarshal([]byte(val), &result)
        return result, nil
    }
    
    // 查数据库
    data, err := queryFunc()
    if err != nil {
        return nil, err
    }
    
    // 写缓存
    jsonVal, _ := json.Marshal(data)
    c.rdb.Set(ctx, key, jsonVal, ttl)
    
    return data, nil
}

func (c *CacheAside) Delete(ctx context.Context, key string) error {
    return c.rdb.Del(ctx, key).Err()
}
```
