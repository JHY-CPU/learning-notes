## Hash Collision Resolution


```javascript
链地址法和开放地址法是解决哈希冲突的两种主要方法。```


```
// 开放地址法（线性探测）
class OpenHash {
  constructor(n=10) { this.keys = new Array(n); this.vals = new Array(n); this.n = n; }
  _h(k) { return k.toString().split('').reduce((a,c)=>a+c.charCodeAt(0),0) % this.n; }
  set(k, v) {
    let i = this._h(k);
    while (this.keys[i] !== undefined && this.keys[i] !== k) i = (i+1) % this.n;
    this.keys[i] = k; this.vals[i] = v;
  }
  get(k) {
    let i = this._h(k);
    while (this.keys[i] !== undefined) { if (this.keys[i] === k) return this.vals[i]; i = (i+1) % this.n; }
    return undefined;
  }
}```


  点击按钮查看结果
