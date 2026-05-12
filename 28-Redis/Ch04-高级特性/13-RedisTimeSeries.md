# RedisTimeSeries

## 一、概念说明

RedisTimeSeries是Redis的时序数据模块，专门用于存储和查询时间序列数据。支持自动聚合、过期策略和标签索引。

## 二、具体用法

### 创建时序

```bash
# 创建时序
TS.CREATE temperature RETENTION 86400 LABELS sensor room1 unit celsius
# RETENTION 86400 - 保留86400秒（24小时）
# LABELS - 标签用于过滤

# 添加数据点
TS.ADD temperature 1704067200 25.5
TS.ADD temperature 1704067260 26.0
TS.ADD temperature 1704067320 25.8

# 自动时间戳
TS.ADD temperature * 25.5
# * 使用当前时间
```

### 查询数据

```bash
# 范围查询
TS.RANGE temperature - + COUNT 10
# - 表示最小时间
# + 表示最大时间

# 按时间范围
TS.RANGE temperature 1704067200 1704067400

# 聚合查询
TS.RANGE temperature - + AGGREGATION avg 60
# 每60秒聚合一次平均值

# 倒序查询
TS.REVRANGE temperature + - COUNT 5
```

### 聚合操作

```bash
# 支持的聚合函数
# avg - 平均值
# sum - 求和
# min - 最小值
# max - 最大值
# range - 范围
# count - 计数
# first - 第一个
# last - 最后一个

TS.RANGE temperature - + AGGREGATION max 300
# 每5分钟取最大值
```

## 三、降采样

```bash
# 创建带降采样的规则
TS.CREATERULE temperature temperature:avg AGGREGATION avg 3600
# 每小时平均值存储到temperature:avg

# 查询降采样数据
TS.RANGE temperature:avg - +
```

## 四、注意事项

1. **时间戳单位**：毫秒级时间戳
2. **内存管理**：及时设置过期策略
3. **标签索引**：用于高效过滤
4. **聚合规则**：自动降采样节省存储
5. **集群支持**：需要相同slot的标签

## 五、Python操作示例

```python
from redis import Redis
from datetime import datetime

r = Redis()

# 创建时序
r.ts().create('cpu:usage', retention_ms=86400000, labels={'host': 'server1', 'metric': 'cpu'})

# 添加数据点
now = int(datetime.now().timestamp() * 1000)
r.ts().add('cpu:usage', now, 75.5)
r.ts().add('cpu:usage', '*', 82.3)  # 自动时间戳

# 批量添加
data_points = [
    (now - 60000, 70.2),
    (now - 30000, 78.5),
    (now, 82.3)
]
for ts, value in data_points:
    r.ts().add('cpu:usage', ts, value)

# 范围查询
results = r.ts().range('cpu:usage', '-', '+')
for ts, value in results:
    print(f"{datetime.fromtimestamp(ts/1000)}: {value}%")

# 聚合查询
avg_results = r.ts().range('cpu:usage', '-', '+', aggregation_type='avg', bucket_size_msec=60000)

# 创建降采样规则
r.ts().createrule('cpu:usage', 'cpu:avg:1h', aggregation_type='avg', bucket_size_msec=3600000)

# 获取最新数据点
latest = r.ts().get('cpu:usage')
print(f"最新: {latest}")

# 多Key聚合
r.ts().mrange('-', '+', filters=['metric=cpu'], aggregation_type='avg', bucket_size_msec=60000)
```

## 六、监控场景应用

```bash
# 场景1：服务器监控
TS.CREATE server:cpu:usage LABELS host server1
TS.CREATE server:memory:usage LABELS host server1
TS.CREATE server:disk:usage LABELS host server1

# 场景2：应用指标
TS.CREATE app:qps LABELS service api
TS.CREATE app:latency LABELS service api
TS.CREATE app:error_rate LABELS service api

# 场景3：业务指标
TS.CREATE biz:orders LABELS type daily
TS.CREATE biz:revenue LABELS type daily
TS.CREATE biz:users:active LABELS type daily

# 场景4：IoT传感器数据
TS.CREATE sensor:temperature LABELS location room1 device sensor1
TS.CREATE sensor:humidity LABELS location room1 device sensor1

# 降采样策略
# 原始数据：每秒1个点，保留24小时
# 1分钟聚合：保留7天
# 1小时聚合：保留30天
# 1天聚合：保留1年
TS.CREATERULE sensor:temperature sensor:temp:avg:1m AGGREGATION avg 60000
TS.CREATERULE sensor:temperature sensor:temp:avg:1h AGGREGATION avg 3600000
TS.CREATERULE sensor:temperature sensor:temp:avg:1d AGGREGATION avg 86400000
```

## 七、性能优化

```bash
# 1. 合理设置保留时间
# 不需要永久保存的数据设置TTL
TS.CREATE temp RETENTION 604800000  # 7天

# 2. 使用降采样减少存储
# 原始数据24小时后删除
# 降采样数据长期保留

# 3. 批量添加
# 使用pipeline批量添加数据点
# 减少网络开销

# 4. 标签设计
# 标签用于过滤，不宜过多
# 每个标签增加内存开销
```
