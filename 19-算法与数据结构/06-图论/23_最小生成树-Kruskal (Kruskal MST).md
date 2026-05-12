# Kruskal MST

  Kruskal 算法按边权重从小到大选择，用并查集判断是否形成环。

## 什么是 Kruskal 算法

  Kruskal 是一种贪心算法，用于求无向连通加权图的最小生成树（MST）。算法将所有边按权重从小到大排序，依次尝试加入每条边。如果加入某条边不会形成环（即两端点不在同一连通分量中），则保留该边。使用并查集（Union-Find）高效判断环。

## 关键性质

    - 贪心策略：每次选择权重最小且不形成环的边
    - 使用并查集的"查找"和"合并"操作，支持路径压缩和按秩合并
    - 最终选出 V-1 条边，构成 MST
    - 适用于边列表表示的图

## 复杂度分析

    - **时间：**O(E log E)，主要是排序的代价
    - **空间：**O(V)，并查集的空间
    - 并查集操作近似 O(1)（反阿克曼函数）

## 适用场景 vs Prim

    - 稀疏图（E 接近 V）：Kruskal 更优，O(E log E)
    - 边列表表示：Kruskal 天然适合
    - 稠密图（E 接近 V^2）：Prim 更优
    - 需要在线/增量构建：Prim 更自然

## 常见陷阱

    - 忘记排序，直接按原序处理边会导致错误结果
    - 并查集未做路径压缩，find 操作可能退化为 O(V)
    - 图不连通时，选出的边数少于 V-1，但仍是"最小生成森林"


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
console.log(kruskal(4, edges)); // cost=6
```


## 实际应用

  在铺设通信线路时，目标是以最低总成本将所有城市连接起来。每个城市是顶点，城市间的线路成本是边权，Kruskal 算法给出最优铺设方案。在聚类分析中，最大边权最小的生成树可用于单链接聚类。
