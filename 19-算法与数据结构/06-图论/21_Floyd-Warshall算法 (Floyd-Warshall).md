## Floyd-Warshall 算法

  多源最短路径算法，使用动态规划思想，O(V^3) 计算所有顶点对之间的最短路径。


```javascript
function floydWarshall(graph) {
  const V = graph.length;
  const dist = graph.map(row => [...row]);
  for (let k = 0; k < V; k++)
    for (let i = 0; i < V; i++)
      for (let j = 0; j < V; j++)
        if (dist[i][j] > dist[i][k] + dist[k][j])
          dist[i][j] = dist[i][k] + dist[k][j];
  return dist;
}```

  ## DP 递推式

  dist[k][i][j] = min(dist[k-1][i][j], dist[k-1][i][k] + dist[k-1][k][j])

  ## 交互演示
