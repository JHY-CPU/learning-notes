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

## 六、热点Key自动发现

```python
import redis
import time
from collections import defaultdict

class HotKeyDetector:
    def __init__(self, threshold=1000, window=60):
        self.r = redis.Redis()
        self.threshold = threshold  # 阈值：每分钟访问次数
        self.window = window
        self.key_counts = defaultdict(int)
        self.hot_keys = set()
    
    def start_monitoring(self):
        """使用MONITOR命令统计热点Key"""
        pubsub = self.r.monitor()
        
        start_time = time.time()
        
        for command in pubsub.listen():
            if time.time() - start_time > self.window:
                # 窗口结束，检查热点Key
                self._check_hot_keys()
                self.key_counts.clear()
                start_time = time.time()
            
            # 统计命令中的Key
            if 'command' in command:
                cmd_parts = command['command'].split()
                if len(cmd_parts) >= 2:
                    key = cmd_parts[1]
                    self.key_counts[key] += 1
    
    def _check_hot_keys(self):
        """检查热点Key"""
        for key, count in self.key_counts.items():
            if count >= self.threshold:
                if key not in self.hot_keys:
                    print(f"发现热点Key: {key}, 访问次数: {count}")
                    self.hot_keys.add(key)
                    self._enable_local_cache(key)
    
    def _enable_local_cache(self, key):
        """启用本地缓存"""
        # 实现热点Key的本地缓存逻辑
        pass

# 使用
detector = HotKeyDetector(threshold=1000, window=60)
detector.start_monitoring()
```

## 七、Key分片详细实现

```python
import redis
import random
import json

r = redis.Redis()

class KeySharding:
    def __init__(self, shard_count=3):
        self.shard_count = shard_count
    
    def get_sharded_key(self, key, shard_id=None):
        """获取分片Key"""
        if shard_id is None:
            shard_id = random.randint(1, self.shard_count)
        return f"{key}:{shard_id}"
    
    def set(self, key, value, ttl=3600):
        """写入所有分片"""
        for i in range(1, self.shard_count + 1):
            sharded_key = f"{key}:{i}"
            r.setex(sharded_key, ttl, json.dumps(value))
    
    def get(self, key):
        """随机读取一个分片"""
        shard_id = random.randint(1, self.shard_count)
        sharded_key = f"{key}:{shard_id}"
        value = r.get(sharded_key)
        return json.loads(value) if value else None
    
    def delete(self, key):
        """删除所有分片"""
        for i in range(1, self.shard_count + 1):
            sharded_key = f"{key}:{i}"
            r.delete(sharded_key)

# 使用
sharding = KeySharding(shard_count=3)

# 写入热点数据（所有分片）
sharding.set("product:hot", {"id": 1, "name": "热门商品", "price": 99.9})

# 读取（随机分片）
product = sharding.get("product:hot")

# 删除（所有分片）
sharding.delete("product:hot")
```

## 八、多级缓存架构详解

```bash
# 完整架构
# 请求 → CDN → Nginx缓存 → 应用本地缓存 → Redis → 数据库
#   ↓
# 每层都可能命中，减少后端压力

# Layer 1: CDN缓存
# 静态资源、热门商品页面
# TTL: 5-30分钟

# Layer 2: Nginx缓存
# API响应缓存
# proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m

# Layer 3: 应用本地缓存
# Guava/Caffeine（Java）
# functools.lru_cache（Python）
# node-cache（Node.js）
# TTL: 30秒-5分钟

# Layer 4: Redis缓存
# 分布式缓存，所有应用实例共享
# TTL: 1-24小时

# Layer 5: 数据库
# 最终数据源

# 热点Key处理
# 热点Key → 本地缓存（TTL 10秒）
# 读写分离 → 读从节点
# Key分片 → 分散到多个Key
```

## 九、热点Key预防策略

```bash
# 1. 业务设计阶段
# - 避免将大量用户数据集中在一个Key
# - 设计合理的数据分片策略
# - 使用Hash存储而非单Key

# 2. 上线前测试
# - 压测识别潜在热点
# - 模拟极端访问模式
# - 验证分片策略有效性

# 3. 运行时监控
# - 实时监控Key访问频率
# - 自动标记热点Key
# - 触发自动保护机制

# 4. 应急预案
# - 热点Key本地缓存
# - 热点Key分片
# - 限流降级
```
