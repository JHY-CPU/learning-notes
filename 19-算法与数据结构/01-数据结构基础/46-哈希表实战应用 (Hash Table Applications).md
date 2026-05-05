## Hash Table Applications


```javascript
哈希表广泛应用于数据库索引、缓存系统、编译器符号表、路由表等。```


```
// 模拟数据库行缓存
class RowCache {
  constructor(ttl=5000) { this.cache = new Map(); this.ttl = ttl; }
  get(id) {
    const row = this.cache.get(id);
    if (!row) return null;
    if (Date.now() > row.expire) { this.cache.delete(id); return null; }
    return row.data;
  }
  set(id, data) { this.cache.set(id, {data, expire: Date.now() + this.ttl}); }
  stats() { return { size: this.cache.size, hitRate: 'N/A' }; }
}```


  点击按钮查看结果
