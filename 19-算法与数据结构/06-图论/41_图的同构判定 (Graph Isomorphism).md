# Graph Isomorphism

  判断两个图是否同构是计算复杂性理论中的重要问题。

## 什么是图同构

  两个图同构意味着它们结构完全相同，只是顶点的命名（编号）不同。形式化地说，存在一个顶点的一一映射，使得边的关系完全保持。图同构问题既未被证明是 P 的，也未被证明是 NP-完全的，属于 NP 中间问题。2015 年 Babai 提出了准多项式时间算法。

## 关键性质

    - 必要条件：顶点数、边数、度序列必须相同
    - Weisfeiler-Lehman 测试是一种高效的启发式方法
    - 对于小图，可以枚举所有顶点排列来验证
    - 实际中大多数图可以通过度序列和邻居度序列快速区分


```
// 图同构的简单判定（不一定完备）
function graphIsomorphismSimple(g1, g2, n) {
  // 必要条件：度序列相同
  const deg1 = new Array(n).fill(0).map((_,i) => (g1[i]||[]).length).sort((a,b)=>a-b);
  const deg2 = new Array(n).fill(0).map((_,i) => (g2[i]||[]).length).sort((a,b)=>a-b);
  for (let i = 0; i < n; i++) if (deg1[i] !== deg2[i]) return false;
  // 还需要进一步验证（Weisfeiler-Lehman 等）
  return '度序列相同，可能同构';
}
console.log(graphIsomorphismSimple({0:[1,2],1:[0],2:[0]}, {0:[1],1:[0,2],2:[1]}, 3));
```


## 复杂度分析

    - **Babai 算法：**准多项式时间 2^O((log n)^c)
    - **度序列检查：**O(V log V)
    - **暴力枚举：**O(V!)，仅适用于小图

## 实际应用

  在化学信息学中，判断两个分子结构图是否同构，用于化合物分类和检索。在模式识别中，图同构用于判断两个拓扑结构是否相同。

## 不变量检查

  在尝试暴力匹配之前，先检查一系列不变量，如果任何不变量不匹配则两个图一定不同构：

  | 不变量 | 检查方法 | 复杂度 |
  | --- | --- | --- |
  | 顶点数 | 直接比较 | O(1) |
  | 边数 | 直接比较 | O(1) |
  | 度序列 | 排序后比较 | O(VlogV) |
  | 度数分布 | 统计各度数出现次数 | O(V) |
  | 连通分量数 | BFS/DFS | O(V+E) |
  | 环的长度 | 检测各长度环 | 较复杂 |

## Weisfeiler-Lehman 测试

```javascript
// Weisfeiler-Lehman 1 维测试（颜色细化）
function wlHash(graph, n) {
  // 初始标签：度数
  let labels = Array.from({length: n}, (_, i) => String((graph[i] || []).length));

  for (let iter = 0; iter < n; iter++) {
    const newLabels = [];
    for (let i = 0; i < n; i++) {
      const neighborLabels = (graph[i] || [])
        .map(j => labels[j])
        .sort()
        .join(',');
      newLabels.push(labels[i] + ':' + neighborLabels);
    }
    // 重编码标签
    const labelMap = new Map();
    let nextId = 0;
    for (const l of newLabels) {
      if (!labelMap.has(l)) labelMap.set(l, nextId++);
    }
    labels = newLabels.map(l => String(labelMap.get(l)));
  }
  // 排序后作为图的哈希
  return labels.sort().join(',');
}

// 如果两个图的 WL 哈希不同，则它们不同构
// 注意：WL 测试对某些图可能误判（如某些强正则图）
```

## 常见陷阱

    - 度序列相同不代表图同构，只是必要条件
    - 简单的启发式方法可能给出假阳性
    - 大规模图的精确同构判定非常困难
    - WL 测试对大多数实际图足够可靠，但存在反例
