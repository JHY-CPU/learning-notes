# 哈密顿路径 (Hamiltonian Path)

  **哈密顿路径：**经过每个顶点恰好一次的路径
**哈密顿回路：**起点=终点的哈密顿路径

## 什么是哈密顿路径

  哈密顿路径要求恰好访问每个顶点一次。哈密顿回路（也称哈密顿圈）是起点和终点重合的哈密顿路径。与欧拉路径不同，哈密顿路径问题是 NP-完全问题，不存在已知的多项式时间解法。

## 关键性质

    - NP-完全问题，n 个顶点有 n! 种排列，暴力搜索代价巨大
    - 没有简单的度数条件能判定哈密顿路径的存在性
    - Dirac 定理：若每个顶点度数 >= n/2，则存在哈密顿回路
    - 小规模问题可用回溯法、状态压缩 DP 解决


> 哈密顿路径问题是 NP-完全问题，目前没有多项式时间解法。通常使用回溯法/暴力搜索。

## 与欧拉路径的对比






  | 欧拉路径 | 哈密顿路径 |
| --- | --- |
| 经过 | 每条边恰好一次 |
| 难度 | 多项式时间可解 |
| 判定 | 度数条件 |

## 复杂度分析

    - **暴力回溯：**O(n!)，阶乘级增长
    - **状态压缩 DP：**O(2^n * n)，适用于 n <= 20
    - **近似算法：**对于大规模问题，常用启发式方法

## 适用场景

    - 旅行商问题（TSP）的变种，寻找经过所有城市的最短回路
    - 棋盘上马的遍历问题（Knight's Tour）
    - 小规模精确求解用状态压缩 DP

## 常见陷阱

    - 不要用 BFS/DFS 直接求哈密顿路径，复杂度不是 O(V+E)
    - 状态压缩 DP 时注意位运算的边界
    - 完全图中任意排列都是哈密顿路径，但一般图中可能不存在

## 状态压缩 DP 实现

```javascript
// 状态压缩 DP 求哈密顿回路（TSP）
function tspDP(dist) {
  const n = dist.length;
  const INF = Infinity;
  // dp[mask][i]: 已访问集合为 mask，当前在 i 的最短路径
  const dp = Array.from({length: 1 << n}, () => new Array(n).fill(INF));
  dp[1][0] = 0;  // 从顶点 0 出发

  for (let mask = 1; mask < (1 << n); mask++) {
    for (let u = 0; u < n; u++) {
      if (!(mask & (1 << u)) || dp[mask][u] === INF) continue;
      for (let v = 0; v < n; v++) {
        if (mask & (1 << v)) continue;  // 已访问
        const newMask = mask | (1 << v);
        dp[newMask][v] = Math.min(dp[newMask][v], dp[mask][u] + dist[u][v]);
      }
    }
  }

  // 回到起点
  let ans = INF;
  const fullMask = (1 << n) - 1;
  for (let u = 0; u < n; u++) {
    ans = Math.min(ans, dp[fullMask][u] + dist[u][0]);
  }
  return ans;
}
```

```cpp
// C++ 状态压缩 DP
int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, 1e9));
    dp[1][0] = 0;
    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u)) || dp[mask][u] >= 1e9) continue;
            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v], dp[mask][u] + dist[u][v]);
            }
        }
    }
    int ans = 1e9;
    for (int u = 0; u < n; u++) ans = min(ans, dp[(1<<n)-1][u] + dist[u][0]);
    return ans;
}
```

## 实际应用

  - **旅行商问题（TSP）：**从起点出发，经过所有城市恰好一次后返回起点，求最短总距离
  - **基因组组装：**DNA 片段的排列顺序问题
  - **电路板钻孔：**最小化钻头在孔之间的移动距离
  - **物流配送：**快递员访问所有客户点的最优路线

## 交互演示：回溯法求解

  在 4 顶点完全图中查找哈密顿回路
