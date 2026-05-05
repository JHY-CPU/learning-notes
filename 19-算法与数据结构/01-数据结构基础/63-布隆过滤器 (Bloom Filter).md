## Bloom Filter


```javascript
布隆过滤器是一种空间高效的集合数据结构，允许一定的假阳性误判。```


```
class BloomFilter {
  constructor(size=100, hashCount=3) {
    this.bits = new Array(size).fill(false);
    this.size = size;
    this.hashCount = hashCount;
  }
  _hash(item, seed) {
    let h = 0;
    for (let i = 0; i < item.length; i++)
      h = (h * 31 + item.charCodeAt(i) + seed) % this.size;
    return h;
  }
  add(item) {
    for (let i = 0; i < this.hashCount; i++)
      this.bits[this._hash(item, i)] = true;
  }
  has(item) {
    for (let i = 0; i < this.hashCount; i++)
      if (!this.bits[this._hash(item, i)]) return false;
    return true;
  }
}```


  点击按钮查看结果
