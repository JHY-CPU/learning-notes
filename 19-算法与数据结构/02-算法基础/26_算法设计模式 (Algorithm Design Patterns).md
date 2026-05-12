# 27-算法设计模式 (Algorithm Design Patterns)

算法设计模式是解决算法问题的通用思想框架，类似于软件设计模式。

## 六大模式

```javascript
const patterns = [
  { name: '暴力枚举', approach: '枚举所有可能', use: '小规模数据', ex: '冒泡排序' },
  { name: '贪心', approach: '局部最优选择', use: '最优子结构', ex: 'Dijkstra/Huffman' },
  { name: '分治', approach: '分解→解决→合并', use: '可分解问题', ex: '归并排序' },
  { name: '动态规划', approach: '子问题+状态转移', use: '重叠子问题', ex: '背包问题' },
  { name: '回溯', approach: '试探+回退', use: '约束满足', ex: 'N皇后' },
  { name: '分支限界', approach: '系统搜索+剪枝', use: 'NP-hard', ex: 'TSP精确解' },
];
```

## 选择指南

| 问题特征 | 推荐模式 |
|----------|---------|
| 求最大/最小值 | 贪心 或 DP |
| 问题可拆为独立子问题 | 分治 |
| 求所有解/组合 | 回溯 |
| 最优子结构 + 无后效性 | 贪心 或 DP |
| 子问题重叠 | DP（记忆化/表格） |
| NP-hard | 分支限界 / 近似 |

## 各模式代码模板

```javascript
// 1. 贪心模板
function greedy(arr) {
  arr.sort(byCriteria); // 按某种规则排序
  let result = 0;
  for (const item of arr) {
    if (canSelect(item)) result += process(item);
  }
  return result;
}

// 2. 分治模板
function divideConquer(problem) {
  if (isSmallEnough(problem)) return solveDirectly(problem);
  const subproblems = split(problem);
  const results = subproblems.map(p => divideConquer(p));
  return merge(results);
}

// 3. 动态规划模板
function dp(n) {
  const dp = new Array(n + 1).fill(0);
  dp[0] = baseCase;
  for (let i = 1; i <= n; i++) {
    dp[i] = transition(dp[i-1], ...);
  }
  return dp[n];
}

// 4. 回溯模板
function backtrack(state, choices, result) {
  if (isSolution(state)) { result.push([...state]); return; }
  for (const choice of choices) {
    if (isValid(state, choice)) {
      state.push(choice);
      backtrack(state, getRemaining(choices, choice), result);
      state.pop();
    }
  }
}

// 5. 分支限界模板
function branchBound(problem) {
  const pq = new MinHeap();
  pq.push({ cost: 0, state: initial });
  while (pq.size()) {
    const { cost, state } = pq.pop();
    if (isSolution(state)) return cost;
    for (const next of getSuccessors(state)) {
      if (next.cost < upperBound) pq.push(next);
    }
  }
}
```

## 模式组合

实际问题往往需要组合多种模式：
- **DP + 贪心**：先用贪心预处理，再用 DP
- **分治 + DP**：分治框架中子问题用 DP 求解
- **回溯 + 剪枝**：用贪心/DP 提供剪枝界限
- **二分 + 贪心**：二分答案 + 贪心验证

## 实际应用

拿到新题后的判断流程：
1. 求最优解？→ 贪心/DP
2. 求所有解？→ 回溯/DFS/BFS
3. 有序查找？→ 二分
4. 问题可分解？→ 分治
5. 求连通性？→ 并查集/BFS/DFS
6. NP-hard？→ 分支限界/近似
