## LFU Cache Design


```javascript
LFU 缓存淘汰最不经常使用的数据，每个元素有使用频率计数。```


```
class LFUCache {
  constructor(cap) {
    this.cap = cap;
    this.cache = new Map(); // key -> {val, freq}
    this.freqMap = new Map(); // freq -> Set
    this.minFreq = 0;
  }
  _updateFreq(key) {
    const {val, freq} = this.cache.get(key);
    this.freqMap.get(freq).delete(key);
    if (this.freqMap.get(freq).size === 0) {
      this.freqMap.delete(freq);
      if (this.minFreq === freq) this.minFreq++;
    }
    this.cache.set(key, {val, freq: freq+1});
    if (!this.freqMap.has(freq+1)) this.freqMap.set(freq+1, new Set());
    this.freqMap.get(freq+1).add(key);
  }
  get(key) {
    if (!this.cache.has(key)) return -1;
    this._updateFreq(key);
    return this.cache.get(key).val;
  }
  // put similar
}```


  点击按钮查看结果
