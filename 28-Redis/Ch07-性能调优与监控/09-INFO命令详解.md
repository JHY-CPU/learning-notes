# INFO命令详解

## 一、概念说明

INFO命令返回Redis服务器的各种信息和统计数据，是监控和调优的重要工具。

## 二、主要Section

```bash
# 服务器信息
INFO server
# redis_version, redis_mode, process_id, tcp_port

# 客户端信息
INFO clients
# connected_clients, blocked_clients, maxclients

# 内存信息
INFO memory
# used_memory, maxmemory, mem_fragmentation_ratio

# 持久化信息
INFO persistence
# rdb_last_bgsave_status, aof_last_rewrite_status

# 统计信息
INFO stats
# total_connections_received, instantaneous_ops_per_sec
# keyspace_hits, keyspace_misses

# 复制信息
INFO replication
# role, connected_slaves, master_link_status

# CPU信息
INFO cpu
# used_cpu_sys, used_cpu_user

# 键空间
INFO keyspace
# db0:keys=1000,expires=500
```

## 三、关键指标

```bash
# 命中率
keyspace_hits / (keyspace_hits + keyspace_misses)

# 内存使用率
used_memory / maxmemory

# 碎片率
mem_fragmentation_ratio

# 每秒操作数
instantaneous_ops_per_sec

# 连接数
connected_clients / maxclients
```

## 四、监控脚本

```python
import redis

r = redis.Redis()

info = r.info()

# 关键指标
print(f"命中率: {info['keyspace_hits']/(info['keyspace_hits']+info['keyspace_misses'])*100:.1f}%")
print(f"内存: {info['used_memory_human']}")
print(f"碎片率: {info['mem_fragmentation_ratio']:.2f}")
print(f"QPS: {info['instantaneous_ops_per_sec']}")
print(f"连接数: {info['connected_clients']}")
```

## 五、注意事项

1. **定期检查**：每天检查关键指标
2. **趋势分析**：记录历史数据，分析趋势
3. **告警设置**：关键指标超过阈值告警
## 六、完整监控脚本

```python
import redis
import json
import time

class RedisInfoMonitor:
    """Redis INFO命令监控工具"""
    
    def __init__(self, host='localhost', port=6379):
        self.r = redis.Redis(host=host, port=port)
    
    def get_all_info(self):
        """获取所有信息"""
        return self.r.info()
    
    def get_section(self, section):
        """获取指定section"""
        return self.r.info(section)
    
    def check_health(self):
        """健康检查"""
        info = self.get_all_info()
        health = {}
        
        # 内存健康
        if info.get('maxmemory', 0) > 0:
            memory_usage = info['used_memory'] / info['maxmemory'] * 100
            health['memory'] = {
                'usage': memory_usage,
                'status': 'ok' if memory_usage < 80 else 'warning'
            }
        
        # 命中率健康
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        if total > 0:
            hit_rate = hits / total * 100
            health['hit_rate'] = {
                'value': hit_rate,
                'status': 'ok' if hit_rate > 80 else 'warning'
            }
        
        # 连接健康
        maxclients = info.get('maxclients', 10000)
        connected = info.get('connected_clients', 0)
        connection_usage = connected / maxclients * 100
        health['connections'] = {
            'connected': connected,
            'max': maxclients,
            'usage': connection_usage,
            'status': 'ok' if connection_usage < 80 else 'warning'
        }
        
        # 碎片率健康
        frag_ratio = info.get('mem_fragmentation_ratio', 1.0)
        health['fragmentation'] = {
            'ratio': frag_ratio,
            'status': 'ok' if 1.0 <= frag_ratio <= 1.5 else 'warning'
        }
        
        # 复制健康
        role = info.get('role', 'unknown')
        if role == 'master':
            slaves = info.get('connected_slaves', 0)
            health['replication'] = {
                'role': 'master',
                'slaves': slaves,
                'status': 'ok' if slaves > 0 else 'warning'
            }
        else:
            link_status = info.get('master_link_status', 'unknown')
            health['replication'] = {
                'role': 'slave',
                'link_status': link_status,
                'status': 'ok' if link_status == 'up' else 'critical'
            }
        
        return health
    
    def print_report(self):
        """打印报告"""
        health = self.check_health()
        
        print("Redis健康检查报告")
        print("=" * 50)
        
        for component, data in health.items():
            status = data['status']
            status_icon = "✓" if status == 'ok' else ("⚠" if status == 'warning' else "✗")
            
            print(f"\n{status_icon} {component.upper()}")
            for key, value in data.items():
                if key != 'status':
                    print(f"  {key}: {value}")
        
        # 检查是否有告警
        warnings = [c for c, d in health.items() if d['status'] != 'ok']
        if warnings:
            print(f"\n告警: {', '.join(warnings)}")
        else:
            print("\n所有指标正常")

# 使用
monitor = RedisInfoMonitor()
monitor.print_report()
```

## 七、INFO各Section详解

```bash
# Server - 服务器信息
redis_version:7.2.0         # Redis版本
redis_mode:standalone        # 运行模式
os:Linux 5.4.0 x86_64       # 操作系统
arch_bits:64                 # 架构
multiplexing_api:epoll       # IO多路复用
process_id:1234              # 进程ID
tcp_port:6379                # 监听端口
uptime_in_seconds:86400      # 运行秒数
uptime_in_days:1             # 运行天数

# Clients - 客户端信息
connected_clients:100        # 连接数
blocked_clients:5            # 阻塞客户端数
maxclients:10000             # 最大连接数
client_recent_max_input_buffer:16384  # 最大输入缓冲区
client_recent_max_output_buffer:0     # 最大输出缓冲区

# Memory - 内存信息
used_memory:1073741824       # 已用内存(bytes)
used_memory_human:1.00G      # 已用内存(可读)
used_memory_peak:1073741824  # 峰值内存
maxmemory:2147483648         # 最大内存
maxmemory_human:2.00G        # 最大内存(可读)
mem_fragmentation_ratio:1.20 # 碎片率
mem_allocator:jemalloc-5.2.1 # 内存分配器

# Persistence - 持久化信息
rdb_last_bgsave_status:ok    # 最后RDB状态
rdb_last_bgsave_time_sec:1   # 最后RDB耗时
aof_last_bgrewrite_status:ok # 最后AOF重写状态
aof_current_size:1048576     # AOF当前大小
aof_base_size:524288         # AOF基础大小

# Stats - 统计信息
total_connections_received:10000  # 总连接数
total_commands_processed:1000000  # 总命令数
instantaneous_ops_per_sec:10000   # 每秒命令数
keyspace_hits:950000              # 命中次数
keyspace_misses:50000             # 未命中次数
rejected_connections:0            # 拒绝连接数

# Replication - 复制信息
role:master                  # 角色
connected_slaves:2           # 连接的从节点
master_link_status:up        # 主节点连接状态
master_last_io_seconds_ago:1 # 最后IO时间
master_sync_in_progress:0    # 是否正在同步

# CPU - CPU信息
used_cpu_sys:10.50           # 系统CPU
used_cpu_user:20.30          # 用户CPU
used_cpu_sys_children:0.00   # 子进程系统CPU
used_cpu_user_children:0.00  # 子进程用户CPU

# Keyspace - 键空间
db0:keys=10000,expires=5000,avg_ttl=3600000
db1:keys=5000,expires=2000,avg_ttl=1800000
```
