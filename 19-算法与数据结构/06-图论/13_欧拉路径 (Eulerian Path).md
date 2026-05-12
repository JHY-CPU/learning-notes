# 欧拉路径 (Eulerian Path)

  **欧拉路径：**经过图中每条边恰好一次的路径（一笔画问题）
**欧拉回路：**起点和终点相同的欧拉路径

## 什么是欧拉路径

  欧拉路径要求恰好访问每条边一次（顶点可以重复访问）。这是经典的"一笔画"问题：能否用一笔画出整个图。欧拉回路是起点和终点重合的欧拉路径。命名来源于数学家欧拉解决的柯尼斯堡七桥问题。

## 关键性质

    - 欧拉路径/回路存在性的判定完全依赖于顶点度数，属于多项式时间问题
    - Hierholzer 算法可在 O(E) 时间内构造欧拉路径
    - 欧拉路径关注"边"，哈密顿路径关注"顶点"
    - 连通性也是必要条件（忽略孤立点后图需连通）

## 判定条件


>
    **无向图：**

    - 欧拉回路：所有顶点度数为偶数

    - 欧拉路径：恰有 0 或 2 个奇度顶点

    **有向图：**

    - 欧拉回路：所有顶点入度=出度

    - 欧拉路径：最多一个顶点出度=入度+1，一个入度=出度+1

## 复杂度分析

    - **判定：**O(V+E)，统计度数并检查连通性
    - **构造：**O(E)，Hierholzer 算法

## 适用场景 vs 哈密顿路径

    - 欧拉路径有多项式解法，哈密顿路径是 NP-完全问题
    - 需要走遍所有"边"时用欧拉路径，走遍所有"顶点"时用哈密顿路径

## 常见陷阱

    - 忽略连通性检查：即使度数条件满足，图不连通也没有欧拉路径
    - 有向图和无向图的判定条件不同，不要混淆
    - 孤立顶点（度数为 0）不影响欧拉路径的存在性

## Hierholzer 算法实现

```javascript
// Hierholzer 算法构造欧拉回路
function hierholzer(edges, n) {
  // 构建邻接表
  const graph = Array.from({length: n}, () => []);
  const degree = new Array(n).fill(0);

  for (const [u, v] of edges) {
    graph[u].push(v);
    graph[v].push(u);  // 无向图
    degree[u]++;
    degree[v]++;
  }

  // 检查所有度数为偶数（欧拉回路条件）
  for (let i = 0; i < n; i++) {
    if (degree[i] % 2 !== 0) return null;  // 不存在欧拉回路
  }

  const path = [];
  const stack = [0];  // 从任意有边的点开始
  while (stack.length) {
    const u = stack[stack.length - 1];
    if (graph[u].length === 0) {
      path.push(stack.pop());
    } else {
      const v = graph[u].pop();
      // 删除反向边
      const idx = graph[v].indexOf(u);
      graph[v].splice(idx, 1);
      stack.push(v);
    }
  }
  return path.reverse();
}
```

```cpp
// C++ Hierholzer
void hierholzer(int u, vector<multiset<int>>& graph, vector<int>& path) {
    while (!graph[u].empty()) {
        int v = *graph[u].begin();
        graph[u].erase(graph[u].begin());
        graph[v].erase(graph[v].find(u));
        hierholzer(v, graph, path);
    }
    path.push_back(u);  // 后序收集
}
```

## 实际应用

  在邮递员问题中，邮递员需要走过每条街道（边）恰好一次，用欧拉路径判断是否可行。在 DNA 测序中，de Bruijn 图的欧拉路径对应原始 DNA 序列的重构。在一笔画游戏中，判断能否用一笔画出整个图形。

## LeetCode 相关题目

  - 332. 重新安排行程（欧拉路径）
  - 753. 破解保险箱（欧拉回路）
  - 2097. 合法重新排列数对
