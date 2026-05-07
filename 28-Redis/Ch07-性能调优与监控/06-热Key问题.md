# 热Key问题

## 一、概念说明

热Key是指访问频率极高的Key，集中在少数节点上，导致该节点负载过高。

## 二、发现热Key

```bash
# 方法1：monitor统计
redis-cli MONITOR | head -n 10000 | awk '{print $4}' | sort | uniq -c | sort -rn

# 方法2：redis-cli --hotkeys（需LFU策略）
redis-cli --hotkeys

# 方法3：客户端统计
# 在应用层统计Key访问频率

# 方法4：proxy统计
# 通过代理层统计
```

## 三、解决方案

```bash
# 1. 本地缓存
# 热Key缓存到应用本地
@Cacheable(value="hot", key="#id")

# 2. Key分片
# product:1001 → product:1001:1, product:1001:2, product:1001:3
shard = random.randint(1, 3)
key = f"product:1001:{shard}"

# 3. 读写分离
# 读操作分散到从节点

# 4. 二级缓存
# 本地缓存 + Redis缓存
```

## 四、注意事项

1. **及时发现**：监控热Key的产生
2. **动态调整**：根据访问模式动态分片
3. **本地缓存一致性**：TTL要短
4. **多层防护**：结合多种方案
