## Combination Sum


```javascript
找到所有和为 target 的组合，数字可以重复使用。```


```
function combinationSum(candidates, target) {
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    if (sum > target) return;
    for (let i = start; i < candidates.length; i++) {
      path.push(candidates[i]);
      backtrack(i, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
console.log(combinationSum([2,3,6,7], 7)); // [[2,2,3],[7]]```


  点击按钮查看结果
