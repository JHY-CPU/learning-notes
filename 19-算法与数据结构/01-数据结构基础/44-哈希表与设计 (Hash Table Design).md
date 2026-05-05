## Hash Table Design


```javascript
设计哈希表时需要考虑哈希函数、冲突解决、动态扩容和负载因子。```


```
// 可扩容的哈希表
class DynamicHashTable {
  constructor() { this.capacity = 4; this.buckets = new Array(this.capacity).fill(null).map(()=>[]); this.size = 0; this.loadFactor = 0.75; }
  _hash(k) { return k.toString().split('').reduce((a,c)=>a+c.charCodeAt(0),0) % this.capacity; }
  set(k,v) {
    const idx = this._hash(k);
    const bucket = this.buckets[idx];
    for (const p of bucket) { if (p[0] === k) { p[1] = v; return; } }
    bucket.push([k,v]); this.size++;
    if (this.size > this.capacity * this.loadFactor) this._resize();
  }
  get(k) { const idx = this._hash(k); for (const p of this.buckets[idx]) { if (p[0] === k) return p[1]; } return undefined; }
  _resize() {
    const old = this.buckets;
    this.capacity *= 2;
    this.buckets = new Array(this.capacity).fill(null).map(()=>[]);
    this.size = 0;
    for (const b of old) for (const [k,v] of b) this.set(k,v);
  }
}```


  点击按钮查看结果
