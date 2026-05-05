## Combination Sum II


```javascript
每个数字只能使用一次，且结果不能重复（需要去重）。```


```
function combinationSum2(candidates, target) {
  candidates.sort((a,b) => a-b);
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    if (sum > target) return;
    for (let i = start; i < candidates.length; i++) {
      if (i > start && candidates[i] === candidates[i-1]) continue;
      path.push(candidates[i]);
      backtrack(i + 1, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
console.log(combinationSum2([10,1,2,7,6,1,5], 8));```


  点击按钮查看结果
