# ZSet范围查询

## 一、概念说明

ZSet提供了强大的范围查询能力，可以按分数范围、排名范围进行查询。支持正序和倒序，以及限制返回数量。

## 二、具体用法

### 按分数范围查询

```bash
# 创建测试数据
ZADD scores 60 "张三" 75 "李四" 85 "王五" 90 "赵六" 95 "钱七"

# 查询分数在70-90之间的
ZRANGEBYSCORE scores 70 90
# 输出: 1) "李四" 2) "王五" 3) "赵六"

# 带分数返回
ZRANGEBYSCORE scores 70 90 WITHSCORES
# 输出: 1) "李四" 2) "75" 3) "王五" 4) "85" 5) "赵六" 6) "90"

# 开区间（不包含边界）
ZRANGEBYSCORE scores (70 (90
# 输出: 1) "李四" 2) "王五" 3) "赵六"
# (70表示>70，不包含70

# 无穷大和无穷小
ZRANGEBYSCORE scores 80 +inf
# 输出: 1) "王五" 2) "赵六" 3) "钱七"

ZRANGEBYSCORE scores -inf 80
# 输出: 1) "张三" 2) "李四" 3) "王五"
```

### 倒序查询

```bash
# 倒序按分数范围
ZREVRANGEBYSCORE scores 90 70
# 输出: 1) "赵六" 2) "王五" 3) "李四"

# 倒序无穷大
ZREVRANGEBYSCORE scores +inf 80
# 输出: 1) "钱七" 2) "赵六" 3) "王五"
```

### 限制返回数量

```bash
# LIMIT offset count
ZRANGEBYSCORE scores 60 100 LIMIT 0 2
# 输出: 1) "张三" 2) "李四"

ZRANGEBYSCORE scores 60 100 LIMIT 2 2
# 输出: 1) "王五" 2) "赵六"

# 倒序LIMIT
ZREVRANGEBYSCORE scores 100 60 LIMIT 0 2
# 输出: 1) "钱七" 2) "赵六"
```

### 按排名范围查询

```bash
# 按排名范围
ZRANGE scores 0 2
# 输出: 1) "张三" 2) "李四" 3) "王五"

# 倒序排名
ZREVRANGE scores 0 1
# 输出: 1) "钱七" 2) "赵六"

# 获取中间排名
ZRANGE scores 1 3
# 输出: 1) "李四" 2) "王五" 3) "赵六"
```

### 范围计数

```bash
# 统计分数范围的数量
ZCOUNT scores 70 90
# 输出: (integer) 3

# 统计排名范围
ZCARD scores
# 输出: (integer) 5
```

### 范围删除

```bash
# 删除分数范围
ZREMRANGEBYSCORE scores 0 70
# 输出: 删除分数<=70的元素

# 删除排名范围
ZREMRANGEBYRANK scores 0 0
# 输出: 删除排名第一的元素
```

## 三、实际应用

```bash
# 分页查询排行榜
# 第1页（每页10条）
ZREVRANGE leaderboard 0 9 WITHSCORES

# 第2页
ZREVRANGE leaderboard 10 19 WITHSCORES

# 分数区间统计
ZCOUNT exam:scores 90 100  # 优秀
ZCOUNT exam:scores 80 89   # 良好
ZCOUNT exam:scores 60 79   # 及格
ZCOUNT exam:scores 0 59    # 不及格
```

## 四、注意事项与常见陷阱

1. **分数是双精度浮点数**：比较可能存在精度问题
2. **开区间语法**：使用(前缀表示不包含
3. **LIMIT只对结果集生效**：先过滤再限制
4. **大范围查询**：查询范围大时注意性能
5. **倒序参数顺序**：ZREVRANGEBYSCORE的min和max需要反过来
