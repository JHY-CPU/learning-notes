# Articulation Points & Bridges

  Tarjan 算法可以找出无向图中的割点和桥（关键边）。

## 什么是割点和桥

  - **割点（Articulation Point）：**删除该顶点及其所有关联边后，图的连通分量数增加
  - **桥（Bridge/关键边）：**删除该边后，图的连通分量数增加

  割点和桥是图的"脆弱点"，在网络可靠性分析中非常重要。

## 关键性质

    - 桥判定：`low[v] > tin[u]`，即 v 的子树无法通过回边回到 u 或其祖先
    - 割点判定（非根）：存在子节点 v 使得 `low[v] >= tin[u]`
    - 根节点是割点当且仅当它有两个或更多子树
    - 一条边是桥的充要条件是它不在任何环中


```
function findBridges(n, edges) {
  const graph = Array.from({length:n}, () => []);
  for (let i = 0; i < edges.length; i++) {
    const [u,v] = edges[i];
    graph[u].push([v, i]); graph[v].push([u, i]);
  }
  let timer = 0;
  const tin = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const bridges = [];
  function dfs(u, pEdge) {
    tin[u] = low[u] = timer++;
    for (const [v, ei] of graph[u]) {
      if (ei === pEdge) continue;
      if (tin[v] !== -1) low[u] = Math.min(low[u], tin[v]);
      else { dfs(v, ei); low[u] = Math.min(low[u], low[v]); if (low[v] > tin[u]) bridges.push([u,v]); }
    }
  }
  for (let i = 0; i < n; i++) if (tin[i] === -1) dfs(i, -1);
  return bridges;
}
console.log(findBridges(5, [[0,1],[0,2],[1,2],[1,3],[3,4]]));
// [[1,3],[3,4]] 或类似的连接
```


## 复杂度分析

    - **时间：**O(V+E)，一次 DFS
    - **空间：**O(V)，tin、low 数组和递归栈

## 适用场景

    - 网络可靠性分析：找出关键节点和关键链路
    - 双连通分量分解的基础
    - 交通网络脆弱性评估

## 常见陷阱

    - 必须处理无向图中的父边（pEdge），避免误判为回边
    - 多个连通分量需分别处理
    - 重边（两点间多条边）不是桥
