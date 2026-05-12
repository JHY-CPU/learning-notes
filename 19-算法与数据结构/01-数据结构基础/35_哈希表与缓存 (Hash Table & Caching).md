# Hash Table & Caching

### 哈希表在缓存中的角色

缓存系统需要快速的键值查找和淘汰策略，哈希表提供 O(1) 的查找能力，结合其他数据结构（双向链表、堆）实现淘汰策略。

### 关键特性

- **LRU（最近最少使用）**：哈希表 + 双向链表，淘汰最久未访问的项
- **LFU（最不经常使用）**：哈希表 + 频率桶，淘汰访问频率最低的项
- **TTL（生存时间）**：哈希表存储过期时间，定期清理

### 时间与空间复杂度

| 缓存类型 | get | put | 空间 |
|---------|-----|-----|------|
| 简单哈希缓存 | O(1) | O(1) | O(cap) |
| LRU Cache | O(1) | O(1) | O(cap) |
| LFU Cache | O(1) | O(1) | O(cap) |

### 适用场景 vs 替代方案

- **热点数据缓存**：LRU 适合访问模式有时间局部性的场景
- **频率敏感缓存**：LFU 适合访问频率相对稳定的场景
- **大规模去重**：布隆过滤器可替代（允许一定误判率）

### 常见陷阱

- JavaScript Map 的遍历顺序是插入顺序，可直接实现简易 LRU
- LRU 在扫描型访问下会频繁淘汰热点数据（可加 TTL 优化）
- 多线程环境下缓存需要加锁或使用无锁结构

```
// LRU 缓存
class LRUCache {
  constructor(cap) { this.cap = cap; this.cache = new Map(); }
  get(key) {
    if (!this.cache.has(key)) return -1;
    const val = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, val);
    return val;
  }
  put(key, val) {
    if (this.cache.has(key)) this.cache.delete(key);
    else if (this.cache.size >= this.cap)
      this.cache.delete(this.cache.keys().next().value);
    this.cache.set(key, val);
  }
}
```


### 实际应用

- **浏览器缓存**：HTTP 缓存用 LRU 策略管理已下载资源
- **CPU 缓存**：硬件缓存行利用哈希定位数据
- **CDN 边缘缓存**：根据 URL 哈希定位最近的缓存节点
- **数据库查询缓存**：MySQL 查询缓存利用哈希表存储 SQL 结果

  点击按钮查看结果
