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
4. **版本选择**：v9是最新版本
