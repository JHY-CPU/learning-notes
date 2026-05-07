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
