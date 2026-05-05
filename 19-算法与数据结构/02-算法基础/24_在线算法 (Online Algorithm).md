## Online Algorithm


```javascript
在线算法逐步处理输入，不能预知未来数据。```


```
// 在线缓存策略
class OnlineCache {
  constructor(cap) { this.cap = cap; this.cache = []; }
  access(page) {
    const idx = this.cache.indexOf(page);
    if (idx >= 0) { this.cache.splice(idx,1); this.cache.push(page); return 'hit'; }
    if (this.cache.length >= this.cap) this.cache.shift();
    this.cache.push(page);
    return 'miss';
  }
  state() { return [...this.cache]; }
}
const cache = new OnlineCache(3);
console.log(cache.access('A')); // miss
console.log(cache.access('B')); // miss
console.log(cache.access('A')); // hit```


  点击按钮查看结果
