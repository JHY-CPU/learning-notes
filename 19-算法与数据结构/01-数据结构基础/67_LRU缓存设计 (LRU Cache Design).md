## LRU Cache Design


```javascript
LRU缓存淘汰最近最少使用的数据，核心是哈希表+双向链表。```


```
class LRUCache {
  constructor(cap) {
    this.cap = cap;
    this.cache = new Map();
  }
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
