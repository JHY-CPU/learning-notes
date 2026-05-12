# 23-近似算法 (Approximation Algorithm)

近似算法针对 NP-hard 问题，在多项式时间内给出有理论保证的近似解。

## 近似比

- 近似比 alpha：近似解/最优解 <= alpha（最小化）或 >= alpha（最大化）
- PTAS：对于任意 eps > 0，存在 (1+eps) 近似的多项式算法

```javascript
// 顶点覆盖的 2-近似算法
function approxVertexCover(edges) {
  const visited = new Set();
  const cover = new Set();

  for (const [u, v] of edges) {
    if (!visited.has(u) && !visited.has(v)) {
      visited.add(u); visited.add(v);
      cover.add(u); cover.add(v);
    }
  }
  return [...cover];
}

// 贪心集合覆盖：ln(n)-近似
function greedySetCover(universe, subsets) {
  const covered = new Set();
  const result = [];

  while (covered.size < universe.length) {
    let bestSet = null, bestGain = 0;
    for (const s of subsets) {
      const gain = s.filter(x => !covered.has(x)).length;
      if (gain > bestGain) { bestGain = gain; bestSet = s; }
    }
    result.push(bestSet);
    bestSet.forEach(x => covered.add(x));
  }
  return result;
}
```

## 经典近似算法

| 问题 | 近似比 | 方法 |
|------|--------|------|
| 顶点覆盖 | 2 | 贪心选边两端点 |
| TSP（三角不等式）| 2 | MST + DFS |
| 集合覆盖 | ln n | 贪心选最大覆盖 |
| 背包 | 1-eps | PTAS |
| 装箱 | 3/2 | First Fit |

## 复杂度

| 算法 | 时间 | 近似比 |
|------|------|--------|
| 顶点覆盖 | O(V+E) | 2 |
| 集合覆盖 | O(n^2) | ln n |
| TSP 近似 | O(V^2) | 2 |

## TSP 2-近似算法

```javascript
// TSP 2-近似（三角不等式成立时）
// 1. 构建 MST
// 2. MST 的 DFS 遍历序即为近似回路
function tspApprox(dist) {
  const n = dist.length;
  // 用 Prim 构建 MST
  const visited = new Set([0]);
  const mst = Array.from({length: n}, () => []);
  const edges = [];

  while (visited.size < n) {
    let minW = Infinity, minU = -1, minV = -1;
    for (const u of visited) {
      for (let v = 0; v < n; v++) {
        if (!visited.has(v) && dist[u][v] < minW) {
          minW = dist[u][v]; minU = u; minV = v;
        }
      }
    }
    visited.add(minV);
    mst[minU].push(minV);
    mst[minV].push(minU);
  }

  // DFS 遍历 MST
  const tour = [];
  const vis = new Set();
  function dfs(u) {
    vis.add(u);
    tour.push(u);
    for (const v of mst[u]) if (!vis.has(v)) dfs(v);
  }
  dfs(0);

  // 计算回路总长度
  let cost = 0;
  for (let i = 0; i < tour.length; i++) {
    cost += dist[tour[i]][tour[(i + 1) % n]];
  }
  return { tour, cost };
}
```

## 何时使用

- 问题被证明是 NP-hard
- 数据规模大，精确算法不可行
- 需要理论保证而非启发式猜测
- 实际场景中近似解足够好

## 常见陷阱

1. **适用条件**：TSP 的 2-近似要求三角不等式
2. **近似比**：某些问题的近似比可能很大
3. **PTAS 未必高效**：虽然多项式时间但常数可能很大
4. **近似比是上界**：实际结果通常比上界好得多
