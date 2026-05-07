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
4. **对比分析**：与基线性能对比
