# Geo地理空间

## 一、概念说明

Redis的Geo功能允许存储地理位置信息（经纬度），并支持距离计算、范围查询等操作。底层使用ZSet实现，分数是GeoHash编码后的值。

## 二、具体用法

### 添加位置

```bash
# 添加单个位置
GEOADD cities 116.40 39.90 "Beijing"
# 输出: (integer) 1

# 添加多个位置
GEOADD cities \
  121.47 31.23 "Shanghai" \
  113.26 23.13 "Guangzhou" \
  114.05 22.55 "Shenzhen" \
  104.06 30.67 "Chengdu"
# 输出: (integer) 4

# 更新位置
GEOADD cities NX 116.41 39.91 "Beijing"
# NX表示仅添加新元素
```

### 距离计算

```bash
# 计算两个位置之间的距离
GEODIST cities Beijing Shanghai
# 输出: "1068192.88"（默认单位：米）

GEODIST cities Beijing Shanghai km
# 输出: "1068.19"（公里）

GEODIST cities Beijing Shanghai mi
# 输出: "663.53"（英里）
```

### 范围查询

```bash
# 以某点为中心查找范围内的位置
GEORADIUS cities 116.40 39.90 1500 km
# 输出: 1) "Beijing" 2) "Shanghai" 3) "Guangzhou"
#       4) "Shenzhen" 5) "Chengdu"

# 带距离返回
GEORADIUS cities 116.40 39.90 1500 km WITHDIST
# 输出: 1) 1) "Beijing" 2) "1.52"
#       2) 1) "Shanghai" 2) "1068.19"
#       ...

# 按距离排序
GEORADIUS cities 116.40 39.90 1500 km WITHDIST ASC

# 限制返回数量
GEORADIUS cities 116.40 39.90 1500 km COUNT 3

# 以已有元素为中心查找
GEORADIUSBYMEMBER cities Beijing 1500 km
# 输出: 以北京为中心1500km内的城市
```

### 获取坐标

```bash
# 获取位置的经纬度
GEOPOS cities Beijing
# 输出: 1) 1) "116.40000134706497"
#             2) "39.90000009167092"

# 获取多个位置
GEOPOS cities Beijing Shanghai Guangzhou
```

### GeoHash

```bash
# 获取GeoHash编码
GEOHASH cities Beijing
# 输出: "wx4g0dtf9e0"

GEOHASH cities Beijing Shanghai
# 输出: 1) "wx4g0dtf9e0"
#       2) "wtw3sjt0wg0"
```

## 三、实际应用

```bash
# 附近的人/店铺
GEOADD stores 116.40 39.90 "store1"
GEOADD stores 116.41 39.91 "store2"
GEOADD stores 116.39 39.89 "store3"

# 查找500米内的店铺
GEORADIUS stores 116.40 39.90 500 m WITHDIST
# 输出附近的店铺及距离

# 打车应用 - 查找附近司机
GEOADD drivers:online 116.40 39.90 "driver1"
GEORADIUS drivers:online 116.40 39.90 3 km COUNT 5
```

## 四、注意事项与常见陷阱

1. **精度限制**：GeoHash精度约0.6米
2. **坐标范围**：经度-180到180，纬度-85到85
3. **底层是ZSet**：可以使用ZREM删除元素
4. **距离计算是球面距离**：不考虑地形
5. **内存占用**：每个位置约52字节
6. **GEORADIUS已废弃**：Redis 6.2+推荐使用GEOSEARCH
