# HyperLogLog详解

## 一、概念说明

HyperLogLog是Redis用于基数统计的数据结构。基数是指集合中不重复元素的个数。HyperLogLog使用极小的内存（固定12KB）来统计大量数据的基数，但有一定的误差（约0.81%）。

## 二、具体用法

### 基本操作

```bash
# 添加元素
PFADD visitors "user1" "user2" "user3"
# 输出: (integer) 1（有新元素添加）

PFADD visitors "user1" "user4"
# 输出: (integer) 1（user4是新元素）

# 获取基数
PFCOUNT visitors
# 输出: (integer) 4（不重复用户数）

# 添加更多
PFADD visitors "user5" "user6" "user7"
PFCOUNT visitors
# 输出: (integer) 7
```

### 合并操作

```bash
# 创建多个HyperLogLog
PFADD page1 "user1" "user2" "user3"
PFADD page2 "user3" "user4" "user5"

# 合并多个HyperLogLog
PFMERGE combined page1 page2
# 输出: OK

# 查看合并后的基数
PFCOUNT combined
# 输出: (integer) 5（去重后）

# 合并并获取基数
PFCOUNT page1 page2
# 输出: (integer) 5（直接合并计算）
```

## 三、实际应用

### 独立访客统计（UV）

```bash
# 每日独立访客
PFADD uv:202401:01 "ip1" "ip2" "ip3" "ip4"
PFADD uv:202401:02 "ip2" "ip3" "ip5" "ip6"

# 查看每日UV
PFCOUNT uv:202401:01
# 输出: (integer) 4

PFCOUNT uv:202401:02
# 输出: (integer) 4

# 合并多日UV
PFMERGE uv:weekly uv:202401:01 uv:202401:02
PFCOUNT uv:weekly
# 输出: (integer) 6（一周独立访客）
```

### 网站流量统计

```bash
# 按小时统计
PFADD traffic:20240101:00 "user1" "user2"
PFADD traffic:20240101:01 "user2" "user3"
PFADD traffic:20240101:02 "user3" "user4"

# 日UV
PFCOUNT traffic:20240101:00 traffic:20240101:01 traffic:20240101:02
```

## 四、误差与内存

```bash
# 误差说明
# - 标准误差：0.81%
# - 实际误差在0.1%-2%之间
# - 数据量越大误差越稳定

# 内存使用
# 固定12KB，与元素数量无关
# 无论添加1个还是100万元素
# 内存占用始终是12KB

# 查看内存使用
MEMORY USAGE visitors
# 输出: 约12KB
```

## 五、注意事项与常见陷阱

1. **有误差**：不适合需要精确计数的场景
2. **不存储元素**：只能统计数量，不能获取具体元素
3. **固定内存**：12KB固定开销，适合海量数据统计
4. **PFMERGE不影响源**：合并操作不删除源数据
5. **不能删除元素**：HyperLogLog不支持删除单个元素
6. **适用场景**：UV统计、独立IP统计等允许误差的场景
