# Redis Sentinel监控

## 一、概念说明

监控Sentinel集群的状态，确保高可用性正常工作。

## 二、Sentinel状态查询

```bash
# 连接Sentinel
redis-cli -p 26379

# 查看主节点信息
SENTINEL master mymaster

# 查看从节点
SENTINEL replicas mymaster

# 查看Sentinel节点
SENTINEL sentinels mymaster

# 获取主节点地址
SENTINEL get-master-addr-by-name mymaster
```

## 三、关键指标

```bash
# 主节点状态
# status: ok
# num-slaves: 2
# num-other-sentinels: 2

# 从节点状态
# master-link-status: up
# master-last-io-seconds-ago: 1

# 故障转移状态
# last-ok-ping-reply: 0
# last-ping-reply: 0
```

## 四、监控脚本

```python
def check_sentinel():
    sentinel = redis.Redis(host='sentinel-host', port=26379)
    
    # 获取主节点
    master = sentinel.execute_command(
        'SENTINEL', 'get-master-addr-by-name', 'mymaster'
    )
    
    # 获取从节点
    replicas = sentinel.execute_command(
        'SENTINEL', 'replicas', 'mymaster'
    )
    
    # 检查状态
    if len(replicas) < 1:
        alert("从节点不足")
```

## 五、注意事项

1. **Sentinel至少3个**：避免脑裂
2. **定期检查**：监控Sentinel本身的状态
3. **故障转移日志**：记录故障转移事件
4. **网络分区**：监控网络分区情况
