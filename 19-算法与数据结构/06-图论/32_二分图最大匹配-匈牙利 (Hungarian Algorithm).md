# Hungarian Algorithm

  匈牙利算法在 O(VE) 时间内求二分图最大匹配。

## 什么是匈牙利算法

  匈牙利算法是基于增广路的思想求二分图最大匹配的经典算法。它依次尝试为每个左部顶点找到匹配对象，若目标已被匹配，则递归地为被匹配者寻找替代对象（即找增广路）。增广路将匹配数增加 1。

## 关键性质

    - 基于 DFS 找增广路：每次 DFS 尝试为一个左部顶点找匹配
    - `seen` 数组防止同一轮 DFS 中重复访问右部顶点
    - 最大匹配数 <= min(|左部|, |右部|)
    - 完美匹配要求 |左部| = |右部| 且最大匹配数 = |左部|


```
function hungarian(graph, n, m) {
  const match = new Array(m).fill(-1);
  function dfs(u, seen) {
    for (const v of (graph[u]||[])) {
      if (seen[v]) continue;
      seen[v] = true;
      if (match[v] === -1 || dfs(match[v], seen)) { match[v] = u; return true; }
    }
    return false;
  }
  let result = 0;
  for (let i = 0; i < n; i++) { const seen = new Array(m).fill(false); if (dfs(i, seen)) result++; }
  return { match, size: result };
}
const graph = {0:[0,1],1:[1,2],2:[2]};
console.log(hungarian(graph, 3, 3)); // size: 3
```


## 复杂度分析

    - **时间：**O(V*E)，每个左部顶点做一次 DFS
    - **空间：**O(V)，match 数组和 seen 数组

## 适用场景 vs 其他匹配算法

    - 二分图匹配首选匈牙利算法，实现简单
    - 一般图匹配需要带花树（Blossom）算法，O(V^3)
    - Hopcroft-Karp 算法 O(E*sqrt(V)) 在稀疏图上更快
    - 最大流方法也能求匹配，但常数较大

## 完整实现

```javascript
// 匈牙利算法完整实现
function hungarianAlgorithm(graph, nLeft, nRight) {
  // matchR[v] = 与右部顶点 v 匹配的左部顶点
  const matchR = new Array(nRight).fill(-1);

  function dfs(u, seen) {
    for (const v of (graph[u] || [])) {
      if (seen[v]) continue;
      seen[v] = true;
      // 如果 v 未匹配，或与 v 匹配的左部顶点能找到替代
      if (matchR[v] === -1 || dfs(matchR[v], seen)) {
        matchR[v] = u;
        return true;
      }
    }
    return false;
  }

  let matchCount = 0;
  for (let u = 0; u < nLeft; u++) {
    const seen = new Array(nRight).fill(false);
    if (dfs(u, seen)) matchCount++;
  }

  // 还原匹配
  const matching = [];
  for (let v = 0; v < nRight; v++) {
    if (matchR[v] !== -1) matching.push([matchR[v], v]);
  }
  return { count: matchCount, matching };
}

// 示例：工人分配任务
const graph = { 0: [0, 1], 1: [1, 2], 2: [0, 2] };
console.log(hungarianAlgorithm(graph, 3, 3));
// { count: 3, matching: [[2,0], [0,1], [1,2]] }
```

```cpp
// C++ 匈牙利算法
bool dfs(int u, vector<vector<int>>& graph, vector<int>& matchR, vector<bool>& seen) {
    for (int v : graph[u]) {
        if (seen[v]) continue;
        seen[v] = true;
        if (matchR[v] == -1 || dfs(matchR[v], graph, matchR, seen)) {
            matchR[v] = u;
            return true;
        }
    }
    return false;
}

int hungarian(vector<vector<int>>& graph, int nLeft, int nRight) {
    vector<int> matchR(nRight, -1);
    int result = 0;
    for (int u = 0; u < nLeft; u++) {
        vector<bool> seen(nRight, false);
        if (dfs(u, graph, matchR, seen)) result++;
    }
    return result;
}
```

## Hopcroft-Karp 算法

```javascript
// Hopcroft-Karp: O(E * sqrt(V))，稀疏图更快
// 核心思想：BFS 找多条最短增广路，同时增广
function hopcroftKarp(graph, nLeft, nRight) {
  const pairU = new Array(nLeft).fill(-1);
  const pairV = new Array(nRight).fill(-1);
  const dist = new Array(nLeft).fill(0);

  function bfs() {
    const queue = [];
    for (let u = 0; u < nLeft; u++) {
      if (pairU[u] === -1) { dist[u] = 0; queue.push(u); }
      else dist[u] = Infinity;
    }
    let found = false;
    while (queue.length) {
      const u = queue.shift();
      for (const v of (graph[u] || [])) {
        const pu = pairV[v];
        if (pu === -1) found = true;
        else if (dist[pu] === Infinity) {
          dist[pu] = dist[u] + 1;
          queue.push(pu);
        }
      }
    }
    return found;
  }

  function dfs(u) {
    for (const v of (graph[u] || [])) {
      const pu = pairV[v];
      if (pu === -1 || (dist[pu] === dist[u] + 1 && dfs(pu))) {
        pairU[u] = v;
        pairV[v] = u;
        return true;
      }
    }
    dist[u] = Infinity;
    return false;
  }

  let matching = 0;
  while (bfs()) {
    for (let u = 0; u < nLeft; u++) {
      if (pairU[u] === -1 && dfs(u)) matching++;
    }
  }
  return matching;
}
```

## 常见陷阱

    - `seen` 数组每轮 DFS 需要重置，否则影响增广路搜索
    - 有向图的匹配需先转为二分图
    - 匹配结果不唯一，但匹配数唯一
    - 匈牙利算法 O(VE) 在稠密图上慢，改用 Hopcroft-Karp

## 实际应用

  - **任务分配：**n 个工人分配 n 个任务，每个工人只能做某些任务
  - **考试安排：**学生和座位的匹配
  - **稳定婚姻：**Gale-Shapley 算法
  - **图像配准：**特征点匹配
