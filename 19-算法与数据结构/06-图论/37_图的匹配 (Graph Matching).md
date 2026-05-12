# Graph Matching

  图的匹配是找边集使任意两条边没有公共顶点，包括最大匹配和完美匹配。

## 匹配的基本概念

  - **匹配（Matching）：**任意两条边没有公共顶点的边子集
  - **最大匹配：**边数最多的匹配
  - **完美匹配：**所有顶点都被匹配的匹配
  - **稳定匹配：**Gale-Shapley 算法求解的配对问题

## 关键性质

    - 二分图最大匹配可用匈牙利算法 O(VE) 求解
    - 一般图最大匹配需要 Edmonds 的带花树算法 O(V^3)
    - 最大匹配数 = 最小顶点覆盖数（König 定理，仅限二分图）
    - 稳定匹配用于双边市场配对（如医学院分配）


```
// 一般图最大匹配（Blossom Algorithm 简述）
// Edmonds 的花算法是第一个多项式时间的一般图匹配算法
// 核心思想：遇到奇环时缩点为"花"
// 时间复杂度 O(V³)
function generalGraphMaxMatching(n, edges) {
  // 简化：对于二分图使用匈牙利
  // 对于一般图需要使用带花树算法
  console.log('一般图匹配使用带花树 (Blossom) 算法');
  return 0;
}
// 二分图最大匹配可使用匈牙利算法
console.log('匹配类型: 最大匹配、完美匹配、稳定匹配');
```


## 复杂度分析

    - **二分图匈牙利：**O(VE)
    - **二分图 Hopcroft-Karp：**O(E*sqrt(V))
    - **一般图 Blossom：**O(V^3)

## 实际应用

  在任务分配中，将工人和任务进行最优配对。在网络交换中，最大匹配决定无冲突的最大数据传输量。稳定匹配用于住院医师分配系统（NRMP）。

## Gale-Shapley 稳定匹配算法

```javascript
// 稳定婚姻问题：Gale-Shapley 算法
function stableMatching(menPrefs, womenPrefs, n) {
  const match = new Array(n).fill(-1);       // women -> men
  const reverseMatch = new Array(n).fill(-1); // men -> women
  const nextProposal = new Array(n).fill(0);  // men 下一个要提议的女性
  const free = new Set(Array.from({length: n}, (_, i) => i));

  while (free.size > 0) {
    const man = free.values().next().value;
    const prefs = menPrefs[man];
    if (nextProposal[man] >= n) break;
    const woman = prefs[nextProposal[man]++];

    if (match[woman] === -1) {
      // 女方空闲
      match[woman] = man;
      reverseMatch[man] = woman;
      free.delete(man);
    } else {
      // 女方已匹配，比较偏好
      const currentMan = match[woman];
      if (womenPrefs[woman].indexOf(man) < womenPrefs[woman].indexOf(currentMan)) {
        // 女方更偏好新追求者
        match[woman] = man;
        reverseMatch[man] = woman;
        reverseMatch[currentMan] = -1;
        free.delete(man);
        free.add(currentMan);
      }
    }
  }
  return reverseMatch;
}
```

## König 定理

  在二分图中：**最大匹配数 = 最小顶点覆盖数**。这一定理将匹配问题与覆盖问题联系起来。

```javascript
// 从最大匹配构造最小顶点覆盖（二分图）
// 1. 找到最大匹配 M
// 2. 从未匹配的左部顶点出发，交替走非匹配边/匹配边
// 3. 左部未访问的 + 右部已访问的 = 最小顶点覆盖
```

## 最大匹配转化为最大流

```javascript
// 二分图匹配 -> 最大流
// 源点 s -> 所有左部顶点（容量 1）
// 左部 -> 右部 有边的对（容量 1）
// 所有右部顶点 -> 汇点 t（容量 1）
// 最大流 = 最大匹配数
```

## 常见陷阱

    - 一般图和二分图的算法不同，不要混用
    - 完美匹配不一定存在，最大匹配一定存在
    - König 定理仅限二分图，一般图不成立
    - Gale-Shapley 算法结果偏向提议方（男方最优/女方最劣）

## LeetCode 相关题目

  - 1595. 连通两组点的最小代价（最小费用最大流）
  - 685. 冗余连接 II（一般图匹配）
  - 871. 最低加油次数（贪心/堆模拟匹配）
