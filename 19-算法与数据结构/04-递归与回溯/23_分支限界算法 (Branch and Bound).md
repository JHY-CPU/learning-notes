## Branch and Bound


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


  点击按钮查看结果
