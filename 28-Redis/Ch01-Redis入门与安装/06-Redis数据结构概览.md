# Redis数据结构概览

## 一、概念说明

Redis支持多种核心数据结构，每种数据结构都有其特定的使用场景和操作命令。了解这些数据结构的特点和底层实现，是高效使用Redis的基础。

### 五大基本类型

| 类型 | 说明 | 底层实现 | 典型场景 |
|------|------|----------|----------|
| String | 字符串/整数 | SDS简单动态字符串 | 缓存、计数器 |
| List | 双向链表 | QuickList | 消息队列、最新列表 |
| Hash | 键值对集合 | HashTable/Ziplist | 用户信息、对象存储 |
| Set | 无序集合 | HashTable/IntSet | 标签、共同好友 |
| ZSet | 有序集合 | SkipList/Ziplist | 排行榜、延迟队列 |

### 特殊类型

```bash
# HyperLogLog - 基数统计（去重计数）
PFADD visitors user1 user2 user3
PFCOUNT visitors
# 输出: (integer) 3

# Geo - 地理位置
GEOADD cities 116.40 39.90 Beijing
GEODIST cities Beijing Shanghai km

# Stream - 消息流（Redis 5.0+）
XADD mystream * field1 value1
XREAD COUNT 2 STREAMS mystream 0
```

## 二、底层数据结构详解

### SDS（Simple Dynamic String）

```bash
# Redis的String类型底层使用SDS
# SDS vs C字符串：
# 1. O(1)获取长度（不用遍历）
# 2. 防止缓冲区溢出
# 3. 二进制安全（可存任何数据）
# 4. 预分配和惰性释放

# 示例
SET mykey "hello"
STRLEN mykey
# 输出: (integer) 5
```

### 压缩列表（Ziplist）

```bash
# 小数据量时使用ziplist节省内存
# 元素个数 < 128 且 元素大小 < 64字节
# 时使用ziplist，否则转为hashtable/skiplist

# 查看编码类型
OBJECT ENCODING mylist
# 输出: "ziplist" 或 "listpack" 或 "quicklist"
```

### 跳表（SkipList）

```bash
# ZSet底层使用跳表实现O(logN)查找
# 跳表是有序的，支持范围查询
# Redis的跳表最多32层

# 验证跳表性能
ZADD leaderboard 100 "player1" 200 "player2" 150 "player3"
ZRANGE leaderboard 0 -1 WITHSCORES
# 输出: 1) "player1" 2) "100.0"
#       3) "player3" 2) "150.0"
#       4) "player2" 3) "200.0"
```

## 三、数据类型选择要点

```bash
# 内存优化原则
# 1. 小数据优先使用ziplist编码
# 2. 整数集合（intset）比hashtable更省内存
# 3. Hash和ZSet在元素少时自动使用ziplist

# 查看实际编码
OBJECT ENCODING key_name

# Hash的ziplist条件
# hash-max-ziplist-entries 128  (元素个数)
# hash-max-ziplist-value 64     (单个值大小)
```

## 四、注意事项与常见陷阱

1. **编码转换是自动的**：当数据量超过阈值时，Redis自动转换编码
2. **String最大512MB**：但实际使用中应控制在KB级别
3. **List元素过多**：超过一定数量后性能下降，考虑分片
4. **Hash适合存储对象**：字段数在128以内时性能最优
5. **ZSet的排名操作是O(logN)**：不是O(1)，大数据量时注意
6. **避免大Key**：单个Key存储过多数据会影响整个Redis性能
