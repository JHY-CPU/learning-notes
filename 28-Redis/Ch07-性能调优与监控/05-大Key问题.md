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
## 六、大Key详细检测脚本

```python
import redis

r = redis.Redis()

def scan_big_keys(threshold_mb=10, threshold_elements=5000):
    """扫描大Key"""
    print(f"扫描大Key (String>{threshold_mb}MB, 集合>{threshold_elements}元素)")
    print("=" * 60)
    
    big_keys = []
    cursor = 0
    total = 0
    
    while True:
        cursor, keys = r.scan(cursor, count=100)
        total += len(keys)
        
        for key in keys:
            key_type = r.type(key).decode()
            memory = r.memory_usage(key) or 0
            
            if key_type == 'string':
                if memory > threshold_mb * 1024 * 1024:
                    big_keys.append({
                        'key': key.decode(),
                        'type': key_type,
                        'memory': memory,
                        'detail': f"大小: {memory/1024/1024:.1f}MB"
                    })
            else:
                # 集合类型
                if key_type == 'hash':
                    elements = r.hlen(key)
                elif key_type == 'list':
                    elements = r.llen(key)
                elif key_type == 'set':
                    elements = r.scard(key)
                elif key_type == 'zset':
                    elements = r.zcard(key)
                else:
                    elements = 0
                
                if elements > threshold_elements:
                    big_keys.append({
                        'key': key.decode(),
                        'type': key_type,
                        'elements': elements,
                        'memory': memory,
                        'detail': f"元素数: {elements}, 大小: {memory/1024:.1f}KB"
                    })
        
        if cursor == 0:
            break
    
    # 输出结果
    print(f"\n扫描完成，共检查 {total} 个Key")
    print(f"发现 {len(big_keys)} 个大Key:\n")
    
    for item in sorted(big_keys, key=lambda x: x['memory'], reverse=True):
        print(f"  {item['key']}")
        print(f"    类型: {item['type']}")
        print(f"    {item['detail']}")
        print()
    
    return big_keys

# 执行扫描
big_keys = scan_big_keys()
```

## 七、大Key异步删除

```python
def safe_delete_large_key(key):
    """安全删除大Key"""
    key_type = r.type(key).decode()
    memory = r.memory_usage(key) or 0
    
    print(f"删除Key: {key}, 类型: {key_type}, 大小: {memory/1024:.1f}KB")
    
    if key_type == 'string':
        # 字符串直接删除
        r.delete(key)
    elif key_type == 'hash':
        # 大Hash分批删除字段
        cursor = 0
        while True:
            cursor, fields = r.hscan(key, cursor, count=100)
            if fields:
                r.hdel(key, *fields)
            if cursor == 0:
                break
        r.delete(key)
    elif key_type == 'list':
        # 大List分批删除
        while r.llen(key) > 0:
            r.ltrim(key, 100, -1)
        r.delete(key)
    elif key_type == 'set':
        # 大Set分批删除
        cursor = 0
        while True:
            cursor, members = r.sscan(key, cursor, count=100)
            if members:
                r.srem(key, *members)
            if cursor == 0:
                break
        r.delete(key)
    elif key_type == 'zset':
        # 大ZSet分批删除
        while r.zcard(key) > 0:
            r.zremrangebyrank(key, 0, 99)
        r.delete(key)
    else:
        # 使用UNLINK异步删除
        r.unlink(key)
    
    print(f"删除完成: {key}")
```

## 八、预防大Key的设计原则

```bash
# 1. 数据分片
# 大Hash → 多个小Hash
# user:1001 → user:1001:shard:0, user:1001:shard:1, ...

# 2. 数据过期
# 为所有Key设置过期时间
# 使用EXPIRE或SETEX命令

# 3. 限制集合大小
# List/Hash/Set/ZSet控制在10000元素以内
# 超过时考虑分片

# 4. 压缩Value
# 使用msgpack/gzip压缩大Value
# 可节省30-50%空间

# 5. 使用合适的数据类型
# 对象 → Hash（ziplist编码更紧凑）
# 简单值 → String
# 列表 → List
# 集合 → Set/ZSet

# 6. 监控Key大小
# 定期使用redis-cli --bigkeys
# 在CI/CD中检查Key大小
```
