# ZSet排名操作

## 一、概念说明

ZSet的排名操作是实现排行榜功能的核心。通过ZRANK/ZREVRANK获取排名，通过ZSCORE获取分数，通过ZINCRBY动态更新分数。

## 二、具体用法

### 排名查询

```bash
# 创建排行榜
ZADD game:rank 5000 "player1" 8000 "player2" 3000 "player3" 9000 "player4"

# 获取排名（从低到高，从0开始）
ZRANK game:rank "player1"
# 输出: (integer) 1

# 获取倒数排名（从高到低）
ZREVRANK game:rank "player1"
# 输出: (integer) 2（第三名）

# 前3名（最高分）
ZREVRANGE game:rank 0 2 WITHSCORES
# 输出: 1) "player4" 2) "9000"
#       3) "player2" 4) "8000"
#       5) "player1" 6) "5000"

# 第N名
ZREVRANGE game:rank 2 2 WITHSCORES
# 输出: 1) "player1" 2) "5000"
```

### 分数操作

```bash
# 获取分数
ZSCORE game:rank "player1"
# 输出: "5000"

# 增加分数
ZINCRBY game:rank 500 "player1"
# 输出: "5500"

# 批量增加分数（使用事务）
MULTI
ZINCRBY game:rank 100 "player1"
ZINCRBY game:rank 200 "player2"
EXEC

# 设置精确分数
ZADD game:rank XX 6000 "player1"
# 输出: (integer) 0
```

### 范围统计

```bash
# 统计分数段人数
ZCOUNT game:rank 5000 10000
# 输出: 分数在5000-10000之间的玩家数量

# 获取指定分数范围的排名
ZRANGEBYSCORE game:rank 5000 10000 WITHSCORES
# 输出: 该分数段的所有玩家

# 获取排名前N%的玩家
ZCARD game:rank
# 输出: (integer) 4
# 前50% = 前2名
ZREVRANGE game:rank 0 1 WITHSCORES
```

### 百分比排名

```bash
# 计算百分比排名
# 假设有100个玩家
ZCARD game:rank
# 输出: 100

# 获取某玩家排名
ZREVRANK game:rank "player50"
# 输出: 49（排名第50）

# 百分比 = (总人数 - 排名) / 总人数 * 100
# (100 - 49) / 100 * 100 = 51%
```

## 三、实际应用

```bash
# 游戏排行榜
ZADD game:daily 1000 "user1"
ZADD game:daily 2000 "user2"
ZINCRBY game:daily 500 "user1"

# 日榜前10
ZREVRANGE game:daily 0 9 WITHSCORES

# 周榜（使用多个ZSet并集）
ZUNIONSTORE game:weekly 7 game:day1 game:day2 game:day3 ...

# 电商销量排行
ZADD sales:202401 1000 "product1" 2000 "product2"
ZREVRANGE sales:202401 0 9 WITHSCORES
```

## 四、注意事项与常见陷阱

1. **排名从0开始**：ZRANK返回0是第一名
2. **分数是浮点数**：可存储整数和浮点数
3. **相同分数的排序**：按成员字符串的字典序
4. **ZINCRBY原子性**：多线程安全
5. **大排行榜性能**：百万级数据排名仍然高效（O(logN)）
6. **定期清理**：定期删除过期的排行榜数据
