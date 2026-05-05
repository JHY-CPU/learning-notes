## Pruning Techniques


```javascript
剪枝是回溯算法的关键优化，通过提前排除不可能的分支加速搜索。```


```
// 剪枝策略总结
// 1. 可行性剪枝：当前路径无法满足条件时提前终止
// 2. 最优性剪枝：当前路径不可能优于已知最优解时终止
// 3. 重复性剪枝：跳过相同元素的重复选择
// 4. 对称性剪枝：利用问题对称性减少搜索空间
// 5. 启发式剪枝：优先探索更有希望的分支

// 示例：带剪枝的组合总和
function combinationSumWithPrune(candidates, target) {
  candidates.sort((a,b) => a-b); // 排序启用剪枝
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    for (let i = start; i < candidates.length; i++) {
      if (sum + candidates[i] > target) break; // 剪枝：排序后更大值无需尝试
      path.push(candidates[i]);
      backtrack(i, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
console.log('剪枝可大幅减少搜索空间');```


  点击按钮查看结果
