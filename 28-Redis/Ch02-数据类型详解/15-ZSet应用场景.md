# ZSet应用场景

## 一、概念说明

ZSet凭借其排序和范围查询能力，在排行榜、延迟队列、时间序列等场景中表现出色。是Redis中最具特色的数据类型之一。

## 二、核心应用场景

### 游戏排行榜

```bash
# 实时排行榜
ZADD game:leaderboard 5000 "player1"
ZADD game:leaderboard 8000 "player2"
ZADD game:leaderboard 3000 "player3"

# 更新分数
ZINCRBY game:leaderboard 1000 "player1"

# 获取前10名
ZREVRANGE game:leaderboard 0 9 WITHSCORES

# 获取玩家排名
ZREVRANK game:leaderboard "player1"
# 输出: 倒数排名（排名越小越靠前）

# 获取玩家分数
ZSCORE game:leaderboard "player1"
```

### 延迟队列

```bash
# 使用分数存储执行时间戳
# 添加延迟任务
ZADD delay:queue 1700000060 "task1"  # 60秒后执行
ZADD delay:queue 1700000120 "task2"  # 120秒后执行
ZADD delay:queue 1700000030 "task3"  # 30秒后执行

# 获取已到期的任务
ZRANGEBYSCORE delay:queue 0 1700000050 LIMIT 0 10
# 输出: 1) "task3"（已到期的任务）

# 消费任务后删除
ZREM delay:queue "task3"

# 循环检查
# 每秒检查一次到期任务
```

### 时间线（Timeline）

```bash
# 使用时间戳作为分数
ZADD timeline:user1 1700000001 "post:1001"
ZADD timeline:user1 1700000002 "post:1002"
ZADD timeline:user1 1700000003 "post:1003"

# 获取最新10条
ZREVRANGE timeline:user1 0 9

# 按时间范围获取
ZRANGEBYSCORE timeline:user1 1700000001 1700000002

# 关注者时间线合并
ZUNIONSTORE timeline:merged 2 timeline:user1 timeline:user2 AGGREGATE MAX
ZREVRANGE timeline:merged 0 9
```

### 滑动窗口限流

```bash
# 记录每次请求的时间戳
ZADD rate:limit:user1001 1700000001 "req1"
ZADD rate:limit:user1001 1700000002 "req2"
ZADD rate:limit:user1001 1700000003 "req3"

# 移除1分钟前的记录
ZREMRANGEBYSCORE rate:limit:user1001 0 1700000000

# 统计窗口内请求数
ZCARD rate:limit:user1001
# 如果 > 100 则拒绝

# 清理旧数据
EXPIRE rate:limit:user1001 60
```

### 成绩排名

```bash
# 学生成绩
ZADD exam:math 95 "张三" 88 "李四" 76 "王五" 92 "赵六"

# 查看排名
ZREVRANGE exam:math 0 -1 WITHSCORES
# 输出: 1) "张三" 2) "95" 3) "赵六" 4) "92"
#       5) "李四" 6) "88" 7) "王五" 8) "76"

# 统计各分数段
ZCOUNT exam:math 90 100  # 优秀
ZCOUNT exam:math 80 89   # 良好
ZCOUNT exam:math 60 79   # 及格
```

## 三、注意事项与常见陷阱

1. **分数是时间戳时**：使用毫秒级时间戳精度更高
2. **延迟队列的定时检查**：需要外部定时任务驱动
3. **排行榜清理**：定期清理过期排行榜数据
4. **分数精度**：浮点数比较可能存在误差
5. **大排行榜分页**：使用LIMIT进行分页查询
6. **并发更新**：ZINCRBY是原子操作，安全并发
