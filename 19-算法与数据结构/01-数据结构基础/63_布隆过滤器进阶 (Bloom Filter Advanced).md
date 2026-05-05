## Bloom Filter Advanced


```javascript
布隆过滤器常用于缓存穿透防护、垃圾邮件过滤、网络爬虫URL去重等。```


```
// 可计数布隆过滤器（支持删除）
class CountingBloomFilter {
  constructor(size=100, hashCount=3) {
    this.counters = new Array(size).fill(0);
    this.hashCount = hashCount;
    this.size = size;
  }
  _hash(item, seed) {
    let h = 0;
    for (let i = 0; i < item.length; i++) h = (h * 31 + item.charCodeAt(i) + seed) % this.size;
    return h;
  }
  add(item) { for (let i = 0; i < this.hashCount; i++) this.counters[this._hash(item, i)]++; }
  remove(item) { for (let i = 0; i < this.hashCount; i++) { const idx = this._hash(item, i); if (this.counters[idx] > 0) this.counters[idx]--; } }
  has(item) { for (let i = 0; i < this.hashCount; i++) if (this.counters[this._hash(item, i)] === 0) return false; return true; }
}```


  点击按钮查看结果
