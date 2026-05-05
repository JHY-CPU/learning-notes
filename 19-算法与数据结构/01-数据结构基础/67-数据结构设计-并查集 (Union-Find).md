## Union-Find


```javascript
并查集用于处理不相交集合的合并与查询，支持路径压缩和按秩合并优化。```


```
class UnionFind {
  constructor(n) { this.parent = Array.from({length:n}, (_,i)=>i); this.rank = new Array(n).fill(0); }
  find(x) {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }
  union(x, y) {
    let px = this.find(x), py = this.find(y);
    if (px === py) return false;
    if (this.rank[px] < this.rank[py]) [px, py] = [py, px];
    this.parent[py] = px;
    if (this.rank[px] === this.rank[py]) this.rank[px]++;
    return true;
  }
  connected(x, y) { return this.find(x) === this.find(y); }
}```


  点击按钮查看结果
