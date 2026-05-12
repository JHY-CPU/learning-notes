# RedisBloom

## 一、概念说明

RedisBloom提供概率数据结构，包括布隆过滤器（Bloom Filter）、Cuckoo Filter、Count-Min Sketch等。用于高效地判断元素是否存在（有极小的误判率）。

## 二、布隆过滤器

### 基本操作

```bash
# 创建布隆过滤器
BF.RESERVE myfilter 0.01 1000
# 0.01 - 误判率1%
# 1000 - 预计元素数量

# 添加元素
BF.ADD myfilter "user1"
BF.ADD myfilter "user2"
# 输出: (integer) 1（成功添加）

# 检查元素是否存在
BF.EXISTS myfilter "user1"
# 输出: (integer) 1（存在）

BF.EXISTS myfilter "user3"
# 输出: (integer) 0（不存在）

# 批量添加
BF.MADD myfilter "user3" "user4" "user5"
```

### 应用场景

```bash
# 1. 防止缓存穿透
# 查询缓存前先检查布隆过滤器
BF.EXISTS cache:users "user:9999"
# 如果返回0，说明数据一定不存在

# 2. 网页去重
BF.ADD visited:url "https://example.com"
BF.EXISTS visited:url "https://example.com"
# 避免重复爬取

# 3. 垃圾邮件过滤
BF.ADD spam:emails "spam@example.com"
```

## 三、其他数据结构

```bash
# Cuckoo Filter - 支持删除
CF.RESERVE mycuckoo 1000
CF.ADD mycuckoo "item1"
CF.EXISTS mycuckoo "item1"
CF.DEL mycuckoo "item1"

# Count-Min Sketch - 频率统计
CMS.INITBYDIM mycms 1000 10
CMS.INCRBY mycms "item1" 5
CMS.QUERY mycms "item1"
# 输出: 频率估计值
```

## 四、注意事项

1. **有误判率**：可能存在假阳性（不存在判为存在）
2. **不支持删除**：标准布隆过滤器不支持删除
3. **内存效率**：比Set节省大量内存
4. **容量规划**：预估准确的元素数量
5. **Redis 4.0+**：需要加载RedisBloom模块

## 五、Python操作RedisBloom

```python
from redis import Redis
from redisbloom.client import Client

rb = Client()

# 布隆过滤器
rb.bfCreate('users', 0.01, 1000000)
rb.bfAdd('users', 'user:1001')
rb.bfMAdd('users', 'user:1002', 'user:1003', 'user:1004')
exists = rb.bfExists('users', 'user:1001')  # True
not_exists = rb.bfExists('users', 'user:9999')  # False

# Cuckoo过滤器
rb.cfCreate('items', 1000000)
rb.cfAdd('items', 'item:1001')
rb.cfExists('items', 'item:1001')  # True
rb.cfDel('items', 'item:1001')
rb.cfExists('items', 'item:1001')  # False

# Count-Min Sketch
rb.cmsInitByDim('freq', 1000, 10)
rb.cmsIncrBy('freq', 'page:1', 5)
rb.cmsIncrBy('freq', 'page:2', 3)
count = rb.cmsQuery('freq', 'page:1')  # [5]

# Top-K
rb.topkReserve('hot', 50, 2000, 7, 0.925)
rb.topkAdd('hot', 'item:1', 'item:2', 'item:1', 'item:3')
top_items = rb.topkList('hot')
```

## 六、布隆过滤器容量规划

```bash
# 容量规划公式
# n = 预期元素数量
# p = 期望误判率
# m = 位数组大小（bits）= -n * ln(p) / (ln2)^2
# k = 哈希函数数量 = m/n * ln2

# 示例
# n = 100万, p = 0.01 (1%)
# m = 1000000 * 4.605 / 0.480 = 9.6M bits ≈ 1.2MB
# k = 9.6M/1M * 0.693 ≈ 7

# 内存使用对比
# 布隆过滤器：1.2MB存储100万元素
# Set：约40MB存储100万元素（40字节/元素）
# 节省约97%的内存
```

## 七、实际应用案例

```bash
# 案例1：防止缓存穿透
# 1. 用户注册时添加到布隆过滤器
BF.ADD user:ids "user:1001"
# 2. 查询时先检查布隆过滤器
BF.EXISTS user:ids "user:9999"
# 返回0表示一定不存在，直接返回

# 案例2：爬虫去重
# 1. 爬取URL时检查是否已访问
BF.ADD visited:urls "https://example.com/page1"
# 2. 如果已存在则跳过
BF.EXISTS visited:urls "https://example.com/page1"

# 案例3：垃圾邮件过滤
# 1. 已知垃圾邮件地址添加到过滤器
BF.ADD spam:emails "spam@example.com"
# 2. 收到邮件时检查
BF.EXISTS spam:emails "new@example.com"

# 案例4：推荐系统去重
# 1. 已推荐内容添加到过滤器
BF.ADD rec:user:1001 "item:2001"
# 2. 生成推荐时过滤已推荐内容
BF.EXISTS rec:user:1001 "item:2002"
```
