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
