# 大Key问题

## 一、概念说明

大Key是指存储了大量数据的Key，会导致Redis操作阻塞、网络拥塞、内存不均等问题。

## 二、发现大Key

```bash
# 方法1：redis-cli --bigkeys
redis-cli --bigkeys
# 输出: 每种数据类型最大的Key

# 方法2：MEMORY USAGE
MEMORY USAGE key_name
# 输出: Key占用的字节数

# 方法3：DEBUG OBJECT
DEBUG OBJECT key_name
# 输出: serializedlength=1024

# 方法4：RDB工具分析
redis-rdb-tools --command memory dump.rdb > memory.csv
```

## 三、大Key标准

```bash
# String类型 > 10MB
# Hash/Set/ZSet > 5000个元素
# List > 5000个元素
```

## 四、解决方案

```bash
# 1. 拆分大Key
# 大Hash → 多个小Hash
HSET user:1001:part1 name "张三"
HSET user:1001:part2 address "北京"

# 2. 分页获取
HSCAN key 0 COUNT 100
LRANGE key 0 99

# 3. 异步删除
UNLINK large_key

# 4. 压缩数据
# 使用msgpack等紧凑格式
```

## 五、注意事项

1. **定期扫描**：定期检查大Key
2. **监控删除**：大Key删除使用UNLINK
3. **拆分策略**：根据业务设计拆分方案
4. **预防为主**：从设计阶段避免大Key
