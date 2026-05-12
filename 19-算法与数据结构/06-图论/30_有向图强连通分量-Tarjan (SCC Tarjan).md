# SCC Tarjan

  Tarjan 算法用一次 DFS 和 lowlink 找出所有强连通分量。

## 什么是强连通分量

  在有向图中，强连通分量（SCC）是顶点的最大子集，其中任意两个顶点之间互相可达。Tarjan 算法通过一次 DFS 找出所有 SCC，是图论中最优美的算法之一。

## 关键性质

    - 使用两个数组：`idx`（DFS 序号）和 `low`（能回溯到的最小序号）
    - 使用栈记录当前 DFS 路径上的顶点
    - 当 `low[u] === idx[u]` 时，u 是一个 SCC 的根，栈中 u 以上所有顶点属于同一 SCC
    - 缩点后（每个 SCC 缩为一个顶点），有向图变成 DAG


```
function tarjan(graph, n) {
  let index = 0;
  const idx = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const onStack = new Array(n).fill(false);
  const stack = [];
  const scc = [];
  function dfs(u) {
    idx[u] = low[u] = index++;
    stack.push(u); onStack[u] = true;
    for (const v of (graph[u]||[])) {
      if (idx[v] === -1) { dfs(v); low[u] = Math.min(low[u], low[v]); }
      else if (onStack[v]) low[u] = Math.min(low[u], idx[v]);
    }
    if (low[u] === idx[u]) {
      const comp = [];
      while (true) { const w = stack.pop(); onStack[w] = false; comp.push(w); if (w === u) break; }
      scc.push(comp);
    }
  }
  for (let i = 0; i < n; i++) if (idx[i] === -1) dfs(i);
  return scc;
}
console.log(tarjan({0:[1],1:[2],2:[0,3],3:[4],4:[3]}, 5));
// [[0,2,1],[4,3]]
```


## 复杂度分析

    - **时间：**O(V+E)，一次 DFS 遍历
    - **空间：**O(V)，idx、low、栈和 onStack 数组

## 适用场景 vs Kosaraju

    - Tarjan 和 Kosaraju 时间复杂度相同，都是 O(V+E)
    - Tarjan 只需一次 DFS，Kosaraju 需要两次
    - Tarjan 代码更紧凑但理解稍难
    - 缩点后得到 DAG，可继续做拓扑排序等

## 常见陷阱

    - `onStack` 标记在弹栈时必须清除，否则影响其他 SCC 的判断
    - `low[u]` 更新时，只考虑 onStack 中的顶点
    - 递归深度过大时需改用迭代实现
