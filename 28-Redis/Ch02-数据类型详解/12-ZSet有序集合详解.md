# ZSet有序集合详解

## 一、概念说明

ZSet（Sorted Set）是Redis特有的数据类型，每个元素都关联一个分数（score），按分数从小到大排序。底层使用SkipList（跳跃表）和Ziplist实现，支持高效的范围查询和排名操作。

## 二、具体用法

### 添加与获取

```bash
# 添加元素（指定分数）
ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
ZADD leaderboard 150 "player3"
# 输出: (integer) 3

# 批量添加
ZADD leaderboard 180 "player4" 120 "player5"
# 输出: (integer) 2

# 获取分数范围的元素（从小到大）
ZRANGE leaderboard 0 -1
# 输出: 1) "player1" 2) "player5" 3) "player3"
#       4) "player4" 5) "player2"

# 获取分数范围的元素（从大到小）
ZREVRANGE leaderboard 0 -1
# 输出: 1) "player2" 2) "player4" 3) "player3"
#       4) "player5" 5) "player1"

# 带分数返回
ZRANGE leaderboard 0 -1 WITHSCORES
# 输出: 1) "player1" 2) "100"
#       3) "player5" 4) "120"
#       ...
```

### 分数范围查询

```bash
# 按分数范围获取
ZRANGEBYSCORE leaderboard 100 180
# 输出: 1) "player1" 2) "player5" 3) "player3" 4) "player4"

# 限制返回数量
ZRANGEBYSCORE leaderboard 100 180 LIMIT 0 2
# 输出: 1) "player1" 2) "player5"

# 无穷大/无穷小
ZRANGEBYSCORE leaderboard 150 +inf
# 输出: 分数>=150的所有元素

ZRANGEBYSCORE leaderboard -inf 150
# 输出: 分数<=150的所有元素
```

### 排名与分数

```bash
# 获取排名（从0开始）
ZRANK leaderboard "player1"
# 输出: (integer) 0（分数最低，排名第0）

ZREVRANK leaderboard "player1"
# 输出: (integer) 4（分数最低，倒数第1）

# 获取分数
ZSCORE leaderboard "player1"
# 输出: "100"

# 获取元素数量
ZCARD leaderboard
# 输出: (integer) 5

# 统计分数范围的元素数量
ZCOUNT leaderboard 100 200
# 输出: (integer) 5
```

### 修改与删除

```bash
# 修改分数
ZADD leaderboard XX 250 "player1"
# 输出: (integer) 0（XX表示仅更新已存在的元素）

# 增加分数
ZINCRBY leaderboard 50 "player1"
# 输出: "300"

# 删除元素
ZREM leaderboard "player5"
# 输出: (integer) 1

# 删除排名范围
ZREMRANGEBYRANK leaderboard 0 0
# 输出: 删除排名最低的元素

# 删除分数范围
ZREMRANGEBYSCORE leaderboard 0 100
# 输出: 删除分数在0-100之间的元素
```

## 三、底层编码

```bash
# 两种编码方式
# 1. ziplist - 元素少且值短时
# 2. skiplist + hashtable - 元素多或值长时

# 阈值配置
# zset-max-ziplist-entries 128
# zset-max-ziplist-value 64

# 查看编码
OBJECT ENCODING leaderboard
# 输出: "skiplist" 或 "ziplist"
```

## 四、注意事项与常见陷阱

1. **分数可重复**：多个元素可以有相同分数
2. **相同分数按字典序**：分数相同时按成员字符串排序
3. **排名从0开始**：ZRANK返回0表示排名第一
4. **O(logN)操作**：大多数操作是O(logN)
5. **WITHSCORES格式**：返回的分数是字符串形式
6. **LIMIT参数**：偏移量和数量，类似SQL的LIMIT
