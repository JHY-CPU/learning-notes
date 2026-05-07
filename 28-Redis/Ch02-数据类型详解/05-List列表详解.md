# List列表详解

## 一、概念说明

Redis的List是有序的双端链表，支持从头部和尾部高效地添加和移除元素。底层使用QuickList实现（Redis 3.2+），结合了ziplist和linkedlist的优点。适用于消息队列、时间线等场景。

## 二、具体用法

### 添加元素

```bash
# 从左边（头部）添加
LPUSH mylist "a" "b" "c"
# 输出: (integer) 3
# 链表顺序: c -> b -> a

# 从右边（尾部）添加
RPUSH mylist "d" "e"
# 输出: (integer) 5
# 链表顺序: c -> b -> a -> d -> e

# 在指定元素前后插入
LINSERT mylist BEFORE "a" "x"
# 输出: (integer) 6
# 链表顺序: c -> b -> x -> a -> d -> e

LINSERT mylist AFTER "a" "y"
# 输出: (integer) 7
```

### 获取元素

```bash
# 获取指定索引的元素
LINDEX mylist 0
# 输出: "c"（第一个元素）

LINDEX mylist -1
# 输出: "e"（最后一个元素）

# 获取指定范围的元素
LRANGE mylist 0 -1
# 输出: 1) "c" 2) "b" 3) "x" 4) "a" 5) "y" 6) "d" 7) "e"

LRANGE mylist 0 2
# 输出: 1) "c" 2) "b" 3) "x"

# 获取列表长度
LLEN mylist
# 输出: (integer) 7
```

### 移除元素

```bash
# 从左边移除
LPOP mylist
# 输出: "c"

# 从右边移除
RPOP mylist
# 输出: "e"

# 移除指定元素
LREM mylist 1 "a"
# 输出: (integer) 1（移除了1个"a"）

# 移除所有匹配的元素
LREM mylist 0 "x"
# 输出: (integer) 1

# 保留指定范围
LTRIM mylist 0 2
# 输出: OK
```

### 修改元素

```bash
# 修改指定索引的元素
LSET mylist 0 "new_value"
# 输出: OK

# 验证修改
LINDEX mylist 0
# 输出: "new_value"
```

## 三、底层编码

```bash
# QuickList = ziplist + linkedlist
# 元素少且小时使用ziplist
# 元素多或大时使用linkedlist

# 配置阈值
# list-max-ziplist-size -2  （单个ziplist最大8KB）
# list-compress-depth 0     （压缩深度）

# 查看编码
OBJECT ENCODING mylist
# 输出: "quicklist"
```

## 四、注意事项与常见陷阱

1. **LPUSH和RPUSH方向**：LPUSH添加到头部，RPUSH添加到尾部
2. **索引从0开始**：0是第一个元素，-1是最后一个
3. **LREM的count参数**：正数从头删，负数从尾删，0删除所有
4. **LTRIM保留范围**：其他元素会被删除
5. **大列表操作**：LRANGE获取大范围可能阻塞，使用分页
6. **QuickList自动优化**：Redis自动管理底层编码
