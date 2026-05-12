# SCC Kosaraju


```javascript
Kosaraju 算法通过两次 DFS 找出有向图的强连通分量。```


```
function kosaraju(graph, n) {
  const visited = new Array(n).fill(false);
  const order = [];
  function dfs1(u) {
    visited[u] = true;
    for (const v of (graph[u]||[])) if (!visited[v]) dfs1(v);
    order.push(u);
  }
  for (let i = 0; i < n; i++) if (!visited[i]) dfs1(i);
  // 反向图
  const rev = Array.from({length:n}, () => []);
  for (const u in graph) for (const v of graph[u]) rev[v].push(Number(u));
  const result = [];
  const visited2 = new Array(n).fill(false);
  function dfs2(u, comp) {
    visited2[u] = true; comp.push(u);
    for (const v of rev[u]) if (!visited2[v]) dfs2(v, comp);
  }
  for (let i = n-1; i >= 0; i--) {
    const u = order[i];
    if (!visited2[u]) { const comp = []; dfs2(u, comp); result.push(comp); }
  }
  return result;
}
const g = {0:[1],1:[2],2:[0,3],3:[4],4:[3]};
console.log(kosaraju(g, 5)); // [[0,2,1],[4,3]]```


## 算法原理

  Kosaraju 算法分两步：
  1. **第一遍 DFS：**对原图做 DFS，记录顶点的完成顺序（后序遍历），入栈
  2. **反向图：**将所有边反向
  3. **第二遍 DFS：**按栈中顺序（逆序）在反向图上做 DFS，每次 DFS 访问的顶点构成一个 SCC

  为什么有效：如果在原图中 u 能到达 v，那么在反向图中 v 能到达 u。第一遍 DFS 确保了从"源 SCC"开始搜索。

## Tarjan 算法对比

```javascript
// Tarjan SCC 算法（只需一遍 DFS）
function tarjanSCC(graph, n) {
  const visited = new Array(n).fill(false);
  const inStack = new Array(n).fill(false);
  const tin = new Array(n).fill(0);
  const low = new Array(n).fill(0);
  const stack = [];
  let timer = 0;
  const result = [];

  function dfs(u) {
    visited[u] = true;
    tin[u] = low[u] = timer++;
    stack.push(u);
    inStack[u] = true;

    for (const v of (graph[u] || [])) {
      if (!visited[v]) {
        dfs(v);
        low[u] = Math.min(low[u], low[v]);
      } else if (inStack[v]) {
        low[u] = Math.min(low[u], tin[v]);
      }
    }

    // u 是 SCC 的根
    if (low[u] === tin[u]) {
      const comp = [];
      let w;
      do {
        w = stack.pop();
        inStack[w] = false;
        comp.push(w);
      } while (w !== u);
      result.push(comp);
    }
  }

  for (let i = 0; i < n; i++) if (!visited[i]) dfs(i);
  return result;
}
```

## 两种算法对比

  | 特性 | Kosaraju | Tarjan |
  | --- | --- | --- |
  | DFS 遍历次数 | 2 次 | 1 次 |
  | 需要反向图 | 是 | 否 |
  | 代码复杂度 | 较简单 | 稍复杂 |
  | 常数因子 | 较大 | 较小 |
  | 扩展性 | 不易扩展 | 可同时求割点/桥 |

## 应用场景

  - **2-SAT 问题：**缩点后判断可满足性
  - **可达性分析：**判断哪些顶点互相可达
  - **社交网络：**发现紧密联系的用户群
  - **编译优化：**识别循环依赖

## LeetCode 相关题目

  - 1192. 查找集群内的关键连接（桥）
  - 323. 无向图中连通分量的数目

  点击按钮查看结果
