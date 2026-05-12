# Redis-benchmark

## 一、概念说明

redis-benchmark是Redis自带的性能基准测试工具，用于测试Redis服务器的性能。

## 二、基本用法

```bash
# 基本测试
redis-benchmark -h 192.168.1.100 -p 6379

# 测试指定命令
redis-benchmark -t set,get -n 100000

# 测试Pipeline
redis-benchmark -t set,get -n 100000 -P 100

# 测试多客户端
redis-benchmark -t set,get -n 100000 -c 50

# 带密码测试
redis-benchmark -a yourpassword -t set -n 100000
```

## 三、输出分析

```bash
# 输出示例
SET: 100000.00 requests per second
GET: 120000.00 requests per second

# 延迟分布
latency <= 1 ms: 95.00%
latency <= 2 ms: 99.00%
latency <= 5 ms: 99.90%
```

## 四、测试场景

```bash
# 1. 单命令性能
redis-benchmark -t set,get,incr,lpush -n 100000

# 2. Pipeline性能
redis-benchmark -t set,get -n 100000 -P 100

# 3. 多客户端并发
redis-benchmark -t set -n 100000 -c 100

# 4. 大Value测试
redis-benchmark -t set -n 10000 -d 102400
# -d 102400: Value大小100KB
```

## 五、注意事项

1. **测试环境**：在专用测试环境测试
2. **避免生产环境**：生产环境测试影响业务
3. **多维度测试**：不同命令、不同Value大小
## 六、完整测试脚本

```bash
#!/bin/bash
# Redis性能基准测试脚本

HOST="192.168.1.100"
PORT="6379"
PASSWORD="yourpassword"

echo "========================================="
echo "Redis性能基准测试"
echo "========================================="

# 1. 基础命令测试
echo -e "\n1. 基础命令测试"
redis-benchmark -h $HOST -p $PORT -a $PASSWORD -t set,get,incr,lpush,lpop -n 100000 -q

# 2. Pipeline测试
echo -e "\n2. Pipeline测试 (100命令/批)"
redis-benchmark -h $HOST -p $PORT -a $PASSWORD -t set,get -n 100000 -P 100 -q

# 3. 并发测试
echo -e "\n3. 并发测试 (50客户端)"
redis-benchmark -h $HOST -p $PORT -a $PASSWORD -t set,get -n 100000 -c 50 -q

# 4. 大Value测试
echo -e "\n4. 大Value测试 (10KB)"
redis-benchmark -h $HOST -p $PORT -a $PASSWORD -t set,get -n 10000 -d 10240 -q

# 5. 混合读写测试
echo -e "\n5. 混合读写测试 (50%读50%写)"
redis-benchmark -h $HOST -p $PORT -a $PASSWORD -t set,get -n 100000 -r 1000000 --csv

echo -e "\n测试完成"
```

## 七、Python测试脚本

```python
import redis
import time
import statistics

r = redis.Redis()

def benchmark_set(n=100000):
    """测试SET性能"""
    start = time.time()
    for i in range(n):
        r.set(f"bench:{i}", f"value:{i}")
    elapsed = time.time() - start
    qps = n / elapsed
    print(f"SET: {n} 次, {elapsed:.2f}秒, {qps:.0f} QPS")
    return qps

def benchmark_get(n=100000):
    """测试GET性能"""
    start = time.time()
    for i in range(n):
        r.get(f"bench:{i}")
    elapsed = time.time() - start
    qps = n / elapsed
    print(f"GET: {n} 次, {elapsed:.2f}秒, {qps:.0f} QPS")
    return qps

def benchmark_pipeline_set(n=100000, batch=100):
    """测试Pipeline SET性能"""
    start = time.time()
    for i in range(0, n, batch):
        pipe = r.pipeline()
        for j in range(batch):
            if i + j < n:
                pipe.set(f"bench:{i+j}", f"value:{i+j}")
        pipe.execute()
    elapsed = time.time() - start
    qps = n / elapsed
    print(f"Pipeline SET: {n} 次, batch={batch}, {elapsed:.2f}秒, {qps:.0f} QPS")
    return qps

def benchmark_pipeline_get(n=100000, batch=100):
    """测试Pipeline GET性能"""
    start = time.time()
    for i in range(0, n, batch):
        pipe = r.pipeline()
        for j in range(batch):
            if i + j < n:
                pipe.get(f"bench:{i+j}")
        pipe.execute()
    elapsed = time.time() - start
    qps = n / elapsed
    print(f"Pipeline GET: {n} 次, batch={batch}, {elapsed:.2f}秒, {qps:.0f} QPS")
    return qps

# 执行测试
print("Redis性能测试")
print("=" * 50)
benchmark_set(10000)
benchmark_get(10000)
benchmark_pipeline_set(10000, batch=100)
benchmark_pipeline_get(10000, batch=100)
```

## 八、测试结果分析

```bash
# 结果解读
# SET: 100000.00 requests per second
# 表示每秒可处理10万次SET操作

# 延迟分布
# latency <= 1 ms: 95.00%
# 95%的请求在1ms内完成

# 影响因素
# 1. 网络延迟（最大影响）
# 2. Value大小
# 3. 并发数
# 4. Redis服务器负载
# 5. 客户端实现

# 对比基线
# 单机Redis: 100K-150K QPS
# Redis集群: 100K-150K QPS * 节数
# Pipeline: 500K-1M QPS

# 性能目标
# 一般业务: >50K QPS
# 高性能要求: >100K QPS
# 极致性能: >500K QPS (Pipeline)
```
