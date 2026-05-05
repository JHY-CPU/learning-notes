## Bloom Filter Variants


```javascript
布隆过滤器有多种变种：可计数、可扩展、压缩、分层等。```


```
// 可扩展布隆过滤器
class ScalableBloomFilter {
  constructor(initialSize=100, growthFactor=2, maxFilters=5) {
    this.filters = [];
    this.growthFactor = growthFactor;
    this.maxFilters = maxFilters;
    this._addFilter(initialSize);
  }
  _addFilter(size) {
    this.filters.push({bits: new Array(size).fill(false), size, count: 0});
  }
  _hash(item, seed, size) {
    let h = 0;
    for (let i = 0; i < item.length; i++) h = (h * 31 + item.charCodeAt(i) + seed) % size;
    return h;
  }
  add(item) {
    const f = this.filters[this.filters.length-1];
    for (let i = 0; i < 3; i++) f.bits[this._hash(item, i, f.size)] = true;
    f.count++;
    if (f.count > f.size * 0.7 && this.filters.length < this.maxFilters)
      this._addFilter(f.size * this.growthFactor);
  }
  has(item) {
    for (const f of this.filters)
      for (let i = 0; i < 3; i++)
        if (!f.bits[this._hash(item, i, f.size)]) return false;
    return true;
  }
}```


  点击按钮查看结果
