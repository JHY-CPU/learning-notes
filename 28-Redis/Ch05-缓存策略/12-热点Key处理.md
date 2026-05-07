# 热点Key处理

## 一、概念说明

热点Key是指访问频率极高的Key，集中在少数节点上，导致该节点负载过高。需要特殊处理来分散压力。

## 二、发现问题

```bash
# 监控热点Key
# 方法1：redis-cli --hotkeys
redis-cli --hotkeys
# 需要maxmemory-policy为LFU

# 方法2：monitor分析
redis-cli MONITOR | head -n 10000 | awk '{print $4}' | sort | uniq -c | sort -rn

# 方法3：客户端统计
# 在客户端统计各Key的访问次数

# 方法4：proxy统计
# 通过代理层统计
```

## 三、解决方案

### 本地缓存

```python
# 热点Key使用本地缓存
from functools import lru_cache

@lru_cache(maxsize=100)
def get_hot_product(product_id):
    return redis.get(f"product:{product_id}")

# 本地缓存优先
# 减少Redis访问
```

### Key分片

```bash
# 将热点Key分散到多个Key
# product:1001 → product:1001:1, product:1001:2, product:1001:3

# 读取时随机选择
import random
shard = random.randint(1, 3)
key = f"product:1001:{shard}"

# 写入时更新所有分片
for i in range(1, 4):
    redis.set(f"product:1001:{i}", data)
```

### 读写分离

```bash
# 热点Key的读操作分散到从节点
# 主节点处理写操作
# 从节点处理读操作

# 客户端配置读写分离
# 写 → Master
# 读 → Slave (负载均衡)
```

## 四、二级缓存架构

```
请求 → 本地缓存 → Redis从节点 → Redis主节点 → DB
         ↓
     热点Key直接返回
```

## 五、注意事项

1. **识别是关键**：准确识别热点Key
2. **本地缓存一致性**：TTL要短
3. **分片策略**：选择合适的分片数
4. **监控效果**：监控热点Key分散效果
5. **动态调整**：根据访问模式动态调整
