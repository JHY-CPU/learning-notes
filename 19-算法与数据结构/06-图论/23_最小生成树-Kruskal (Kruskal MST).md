## Kruskal MST


```javascript
Kruskal 算法按边权重从小到大选择，用并查集判断是否形成环。```


```
class UnionFind {
  constructor(n) { this.parent = Array.from({length:n},(_,i)=>i); this.rank = new Array(n).fill(0); }
  find(x) { if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]); return this.parent[x]; }
  union(x,y) { let px=this.find(x),py=this.find(y); if(px===py)return false; if(this.rank[px] a[2] - b[2]); // 按权重排序
  const uf = new UnionFind(n);
  const mst = [];
  let cost = 0;
  for (const [u,v,w] of edges) {
    if (uf.union(u, v)) { mst.push([u,v,w]); cost += w; }
  }
  return { mst, cost };
}
const edges = [[0,1,4],[0,2,3],[1,2,1],[1,3,2],[2,3,4]];
console.log(kruskal(4, edges)); // cost=6```


  点击按钮查看结果
