## Hash Table & Caching


```javascript
哈希表是缓存系统的核心实现基础，如 LRU 缓存结合哈希表和双向链表。```


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
}```


  点击按钮查看结果
