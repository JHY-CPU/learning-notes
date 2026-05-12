# Branch and Bound


```javascript
分支限界通过上界/下界函数剪枝，常用于组合优化问题。```


```
// 分支限界解决旅行商问题（TSP）
// 维护当前最优解的上界，剪掉下界大于上界的分支
function tspBranchBound(dist) {
  const n = dist.length;
  let best = Infinity;
  const visited = new Array(n).fill(false);
  function bound(path, cur) {
    // 简化下界：当前路径长度 + 剩余节点的最小边和
    let b = cur;
    for (let i = 0; i < n; i++)
      if (!visited[i]) b += Math.min(...dist[i].filter((_,j)=>!visited[j] || j===0));
    return b;
  }
  function backtrack(path, cur, count) {
    if (count === n) {
      best = Math.min(best, cur + dist[path[path.length-1]][0]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      const newCur = cur + dist[path[path.length-1]][i];
      if (newCur >= best) continue; // 分支限界剪枝
      visited[i] = true;
      path.push(i);
      backtrack(path, newCur, count+1);
      path.pop();
      visited[i] = false;
    }
  }
  visited[0] = true;
  backtrack([0], 0, 1);
  return best;
}
console.log('分支限界大幅减少搜索空间');```


## 分支限界 vs 回溯法

  | 特性 | 回溯法 | 分支限界 |
  | --- | --- | --- |
  | 目标 | 找所有解 | 找最优解 |
  | 搜索策略 | 深度优先 | BFS/优先队列 |
  | 剪枝依据 | 约束条件 | 上界/下界函数 |
  | 适用 | 组合枚举 | 组合优化 |

## 0-1 背包分支限界

```javascript
// 0-1 背包问题：分支限界法（优先队列）
function knapsackBB(weights, values, capacity) {
  const n = weights.length;
  // 按价值密度排序
  const items = Array.from({length: n}, (_, i) => ({
    i, ratio: values[i] / weights[i]
  })).sort((a, b) => b.ratio - a.ratio);

  function bound(idx, curWeight, curValue) {
    let w = curWeight, v = curValue;
    for (let i = idx; i < n; i++) {
      const j = items[i].i;
      if (w + weights[j] <= capacity) {
        w += weights[j];
        v += values[j];
      } else {
        // 分数背包计算上界
        v += (capacity - w) * items[i].ratio;
        break;
      }
    }
    return v;
  }

  let best = 0;
  // 优先队列：[上界, 当前价值, 当前重量, 当前索引]
  const pq = [[bound(0, 0, 0), 0, 0, 0]];

  while (pq.length) {
    pq.sort((a, b) => b[0] - a[0]);
    const [ub, val, w, idx] = pq.shift();
    if (ub <= best) continue;  // 剪枝
    if (idx >= n) { best = Math.max(best, val); continue; }
    const j = items[idx].i;
    // 选第 idx 个物品
    if (w + weights[j] <= capacity) {
      const newW = w + weights[j], newV = val + values[j];
      best = Math.max(best, newV);
      const newUb = bound(idx + 1, newW, newV);
      if (newUb > best) pq.push([newUb, newV, newW, idx + 1]);
    }
    // 不选第 idx 个物品
    const noUb = bound(idx + 1, w, val);
    if (noUb > best) pq.push([noUb, val, w, idx + 1]);
  }
  return best;
}
console.log(knapsackBB([2,3,4,5], [3,4,5,6], 8)); // 10
```

## 上界函数设计原则

  - 上界必须 >= 实际最优解（用于最大化问题）
  - 下界必须 <= 实际最优解（用于最小化问题）
  - 上界/下界计算越紧致，剪枝效果越好
  - 常用方法：松弛约束（如分数背包代替 0-1 背包）

## 实际应用

  - **旅行商问题：**当前路径长度 + 剩余最小边权作为下界
  - **调度问题：**贪心调度的结果作为上界
  - **整数规划：**线性规划松弛提供界

  点击按钮查看结果
