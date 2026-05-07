# Set集合运算

## 一、概念说明

Redis的Set支持丰富的集合运算，包括交集（SINTER）、并集（SUNION）和差集（SDIFF）。这些运算是原子性的，可以高效地处理集合之间的关系。

## 二、具体用法

### 准备数据

```bash
# 创建测试数据
SADD setA "a" "b" "c" "d"
SADD setB "c" "d" "e" "f"
SADD setC "d" "e" "f" "g"
```

### 交集（SINTER）

```bash
# 获取交集
SINTER setA setB
# 输出: 1) "c" 2) "d"（setA和setB共有的元素）

# 多集合交集
SINTER setA setB setC
# 输出: 1) "d"（三个集合共有的元素）

# 交集元素数量
SINTERCARD 2 setA setB
# 输出: (integer) 2（Redis 7.0+）

# 交集结果存入新集合
SINTERSTORE result:setA:setB setA setB
# 输出: (integer) 2
SMEMBERS result:setA:setB
# 输出: 1) "c" 2) "d"
```

### 并集（SUNION）

```bash
# 获取并集
SUNION setA setB
# 输出: 1) "a" 2) "b" 3) "c" 4) "d" 5) "e" 6) "f"

# 多集合并集
SUNION setA setB setC
# 输出: 1) "a" 2) "b" 3) "c" 4) "d" 5) "e" 6) "f" 7) "g"

# 并集结果存入新集合
SUNIONSTORE result:all setA setB setC
# 输出: (integer) 7
```

### 差集（SDIFF）

```bash
# 获取差集（从第一个集合减去后面的）
SDIFF setA setB
# 输出: 1) "a" 2) "b"（在setA中但不在setB中）

SDIFF setB setA
# 输出: 1) "e" 2) "f"（在setB中但不在setA中）

# 多集合差集
SDIFF setA setB setC
# 输出: 1) "a" 2) "b"（在setA中但不在setB和setC中）

# 差集结果存入新集合
SDIFFSTORE result:setA-only setA setB
# 输出: (integer) 2
```

## 三、实际应用

```bash
# 共同好友
SADD user:1001:friends "user2" "user3" "user4"
SADD user:1002:friends "user3" "user4" "user5"
SINTER user:1001:friends user:1002:friends
# 输出: 1) "user3" 2) "user4"

# 推荐好友（差集）
SDIFF user:1002:friends user:1001:friends
# 输出: 1) "user5"（你可能认识的人）

# 用户兴趣交集
SADD user:1001:interests "Java" "Redis" "Linux"
SADD user:1002:interests "Redis" "Python" "Linux"
SINTER user:1001:interests user:1002:interests
# 输出: 1) "Redis" 2) "Linux"
```

## 四、注意事项与常见陷阱

1. **SDIFF顺序重要**：差集结果取决于第一个集合
2. **STORE版本**：SINTERSTORE/SUNIONSTORE/SDIFFSTORE将结果存储
3. **原子性**：集合运算是原子的，不会被中断
4. **性能考虑**：大集合的交集运算可能较慢
5. **SINTERCARD限制**：Redis 7.0+才支持
6. **内存使用**：STORE版本会占用额外内存
