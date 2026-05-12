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
## 六、Sentinel详细监控

```python
import redis
import time

class SentinelMonitor:
    """Sentinel监控工具"""
    
    def __init__(self, sentinel_host='localhost', sentinel_port=26379):
        self.sentinel = redis.Redis(host=sentinel_host, port=sentinel_port)
    
    def get_masters(self):
        """获取所有主节点"""
        return self.sentinel.execute_command('SENTINEL', 'masters')
    
    def get_master_info(self, master_name):
        """获取主节点详细信息"""
        info = self.sentinel.execute_command('SENTINEL', 'master', master_name)
        result = {}
        for i in range(0, len(info), 2):
            result[info[i].decode()] = info[i+1].decode()
        return result
    
    def get_slaves(self, master_name):
        """获取从节点列表"""
        slaves = self.sentinel.execute_command('SENTINEL', 'replicas', master_name)
        result = []
        for slave in slaves:
            slave_info = {}
            for i in range(0, len(slave), 2):
                slave_info[slave[i].decode()] = slave[i+1].decode()
            result.append(slave_info)
        return result
    
    def get_sentinels(self, master_name):
        """获取Sentinel节点列表"""
        sentinels = self.sentinel.execute_command('SENTINEL', 'sentinels', master_name)
        result = []
        for s in sentinels:
            s_info = {}
            for i in range(0, len(s), 2):
                s_info[s[i].decode()] = s[i+1].decode()
            result.append(s_info)
        return result
    
    def get_master_addr(self, master_name):
        """获取主节点地址"""
        addr = self.sentinel.execute_command(
            'SENTINEL', 'get-master-addr-by-name', master_name
        )
        return (addr[0].decode(), int(addr[1])) if addr else None
    
    def check_health(self, master_name):
        """检查健康状态"""
        info = self.get_master_info(master_name)
        
        health = {
            'master_name': master_name,
            'status': info.get('flags', 'unknown'),
            'num_slaves': int(info.get('num-slaves', 0)),
            'num_sentinels': int(info.get('num-other-sentinels', 0)) + 1,
            'quorum': int(info.get('quorum', 0)),
            'last_ok_ping': int(info.get('last-ok-ping-reply', 0)),
        }
        
        # 判断健康状态
        if 's_down' in health['status']:
            health['health'] = 'critical'
        elif 'o_down' in health['status']:
            health['health'] = 'critical'
        elif health['num_slaves'] < 1:
            health['health'] = 'warning'
        else:
            health['health'] = 'ok'
        
        return health
    
    def print_report(self, master_name='mymaster'):
        """打印报告"""
        health = self.check_health(master_name)
        
        print(f"Sentinel监控报告 - {master_name}")
        print("=" * 50)
        print(f"状态: {health['status']}")
        print(f"健康: {health['health']}")
        print(f"从节点数: {health['num_slaves']}")
        print(f"Sentinel数: {health['num_sentinels']}")
        print(f"法定人数: {health['quorum']}")
        
        slaves = self.get_slaves(master_name)
        if slaves:
            print("\n从节点:")
            for slave in slaves:
                print(f"  {slave.get('ip')}:{slave.get('port')} - {slave.get('flags', 'unknown')}")

# 使用
monitor = SentinelMonitor(sentinel_host='192.168.1.100', sentinel_port=26379)
monitor.print_report('mymaster')
```

## 七、Sentinel故障转移监控

```bash
# 监控故障转移事件
SENTINEL ckquorum mymaster
# 检查Sentinel法定人数是否足够

# 监控配置纪元
SENTINEL master mymaster | grep config-epoch
# config-epoch: 1
# 每次故障转移纪元递增

# 监控最后切换时间
SENTINEL master mymaster | grep last-hello-message
# 最后收到hello消息的时间

# 强制故障转移（慎用）
SENTINEL failover mymaster
# 手动触发故障转移

# 重置Sentinel
SENTINEL reset mymaster
# 重置Sentinel状态

# 监控脚本
def monitor_failover_events(sentinel):
    pubsub = sentinel.pubsub()
    pubsub.subscribe('+switch-master')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"故障转移事件: {message['data']}")
```
