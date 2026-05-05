## Permutations


```javascript
给定不含重复数字的数组，返回所有可能的全排列。```


```
function permute(nums) {
  const res = [];
  function backtrack(path, used) {
    if (path.length === nums.length) { res.push([...path]); return; }
    for (let i = 0; i < nums.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      path.push(nums[i]);
      backtrack(path, used);
      path.pop();
      used[i] = false;
    }
  }
  backtrack([], new Array(nums.length).fill(false));
  return res;
}
console.log(permute([1,2,3])); // 6个排列```


  点击按钮查看结果
