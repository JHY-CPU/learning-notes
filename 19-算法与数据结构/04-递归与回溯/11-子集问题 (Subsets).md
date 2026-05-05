## Subsets


```javascript
求集合的所有子集/幂集，是回溯的入门问题。```


```
function subsets(nums) {
  const res = [];
  function backtrack(start, path) {
    res.push([...path]);
    for (let i = start; i < nums.length; i++) {
      path.push(nums[i]);
      backtrack(i + 1, path);
      path.pop();
    }
  }
  backtrack(0, []);
  return res;
}
console.log(subsets([1,2,3]));
// [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]```


  点击按钮查看结果
