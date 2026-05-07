# Set集合详解

## 一、概念说明

Redis的Set是String类型的无序不重复集合。支持添加、删除、查找元素以及集合运算（交集、并集、差集）。底层使用HashTable或IntSet实现。

## 二、具体用法

### 基本操作

```bash
# 添加元素
SADD myset "a" "b" "c" "d"
# 输出: (integer) 4（添加了4个新元素）

SADD myset "a" "e"
# 输出: (integer) 1（"a"已存在，只添加了"e"）

# 获取所有元素
SMEMBERS myset
# 输出: 1) "a" 2) "b" 3) "c" 4) "d" 5) "e"

# 获取集合大小
SCARD myset
# 输出: (integer) 5

# 检查元素是否存在
SISMEMBER myset "a"
# 输出: (integer) 1（存在）

SISMEMBER myset "z"
# 输出: (integer) 0（不存在）
```

### 删除与随机获取

```bash
# 删除元素
SREM myset "e"
# 输出: (integer) 1

SREM myset "z"
# 输出: (integer) 0（不存在）

# 随机获取元素
SRANDMEMBER myset 2
# 输出: 1) "b" 2) "d"（随机2个）

SRANDMEMBER myset -2
# 输出: 允许重复的随机2个

# 随机弹出元素
SPOP myset
# 输出: "c"（随机移除一个）

SPOP myset 2
# 输出: 1) "a" 2) "b"（随机移除2个）
```

### 遍历集合

```bash
# SSCAN安全遍历
SADD largeset "item1" "item2" "item3" "item4" "item5"
SSCAN largeset 0 MATCH item* COUNT 2
# 输出: 1) "3"（游标）
#        2) 1) "item1" 2) "item2"
```

## 三、底层编码

```bash
# 两种编码方式
# 1. intset（整数集合）- 元素都是整数且数量少
# 2. hashtable（哈希表）- 元素包含非整数或数量多

# 阈值配置
# set-max-intset-entries 512

# 查看编码
SADD numset 1 2 3
OBJECT ENCODING numset
# 输出: "intset"

SADD strset "a" "b" "c"
OBJECT ENCODING strset
# 输出: "hashtable"
```

## 四、应用场景

```bash
# 1. 标签系统
SADD user:1001:tags "Java" "Redis" "Linux"

# 2. 去重
SADD visitors:202401 "ip1" "ip2" "ip3"
SCARD visitors:202401
# 输出: 独立访客数

# 3. 抽奖
SADD lottery "user1" "user2" "user3" "user4" "user5"
SRANDMEMBER lottery 3
# 随机抽取3名中奖者

# 4. 权限集合
SADD user:1001:permissions "read" "write" "delete"
SISMEMBER user:1001:permissions "admin"
# 检查是否有管理员权限
```

## 五、注意事项与常见陷阱

1. **无序性**：SMEMBERS返回顺序不确定
2. **intset自动转换**：添加非整数元素时转为hashtable
3. **SPOP不可恢复**：弹出的元素被删除
4. **SRANDMEMBER不删除元素**：只是随机获取
5. **SMEMBERS大集合**：大集合使用SSCAN遍历
6. **集合运算是O(N)**：大集合的交并差操作可能较慢
