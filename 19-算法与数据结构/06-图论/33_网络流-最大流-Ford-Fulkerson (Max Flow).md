# Max Flow


```javascript
Ford-Fulkerson 方法通过增广路径不断找增广路求最大流。```


```
// 简化版 Ford-Fulkerson
function maxFlow(capacity, source, sink) {
  const n = capacity.length;
  const residual = capacity.map(r => [...r]);
  let flow = 0;
  function bfs(parent) {
    const visited = new Array(n).fill(false);
    const q = [source]; visited[source] = true; parent[source] = -1;
    while (q.length) {
      const u = q.shift();
      for (let v = 0; v < n; v++) {
        if (!visited[v] && residual[u][v] > 0) {
          parent[v] = u; visited[v] = true; q.push(v);
        }
      }
    }
    return visited[sink];
  }
  const parent = new Array(n);
  while (bfs(parent)) {
    let pathFlow = Infinity;
    for (let v = sink; v !== source; v = parent[v]) pathFlow = Math.min(pathFlow, residual[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) { residual[parent[v]][v] -= pathFlow; residual[v][parent[v]] += pathFlow; }
    flow += pathFlow;
  }
  return flow;
}
const cap = [[0,16,13,0,0,0],[0,0,10,12,0,0],[0,4,0,0,14,0],[0,0,9,0,0,20],[0,0,0,7,0,4],[0,0,0,0,0,0]];
console.log(maxFlow(cap, 0, 5)); // 23```


## 核心概念

  - **流网络：**有向图，每条边有容量限制，源点 s 只有出边，汇点 t 只有入边
  - **流：**每条边上的流量不超过容量，且除 s 和 t 外每个顶点流入=流出（流量守恒）
  - **残余容量：**`c_f(u,v) = c(u,v) - f(u,v)`，即还能流多少
  - **反向边：**允许"反悔"，`c_f(v,u) = c(v,u) + f(u,v)`
  - **增广路：**从 s 到 t 的一条所有残余容量 > 0 的路径
  - **最大流最小割定理：**最大流 = 最小割的容量

## Ford-Fulkerson 方法

```javascript
// Ford-Fulkerson (DFS 找增广路)
function fordFulkerson(capacity, source, sink) {
  const n = capacity.length;
  const residual = capacity.map(r => [...r]);
  let flow = 0;

  function dfs(u, minCap, visited) {
    if (u === sink) return minCap;
    visited[u] = true;
    for (let v = 0; v < n; v++) {
      if (!visited[v] && residual[u][v] > 0) {
        const pushed = dfs(v, Math.min(minCap, residual[u][v]), visited);
        if (pushed > 0) {
          residual[u][v] -= pushed;
          residual[v][u] += pushed;
          return pushed;
        }
      }
    }
    return 0;
  }

  while (true) {
    const visited = new Array(n).fill(false);
    const pushed = dfs(source, Infinity, visited);
    if (pushed === 0) break;
    flow += pushed;
  }
  return flow;
}
```

## 最小割提取

```javascript
// 在最大流后的残余图中，从 s 做 BFS/DFS
// 能到达的顶点集合为 S，不能到达的为 T
// 最小割 = 所有从 S 到 T 的边的容量之和
function minCut(residual, source, n) {
  const visited = new Array(n).fill(false);
  const queue = [source];
  visited[source] = true;
  while (queue.length) {
    const u = queue.shift();
    for (let v = 0; v < n; v++) {
      if (!visited[v] && residual[u][v] > 0) {
        visited[v] = true;
        queue.push(v);
      }
    }
  }
  return visited;  // visited 为 true 的顶点属于 S
}
```

## 应用场景

  - **网络带宽：**求两节点间最大数据传输率
  - **图像分割：**最小割用于前景/背景分离
  - **二分图匹配：**转化为最大流问题
  - **项目选择：**最大权闭合子图问题

  点击按钮查看结果
