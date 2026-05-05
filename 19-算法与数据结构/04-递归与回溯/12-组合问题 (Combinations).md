## Combinations


```javascript
从 n 个数中选 k 个的所有组合。```


```
function combine(n, k) {
  const res = [];
  function backtrack(start, path) {
    if (path.length === k) { res.push([...path]); return; }
    for (let i = start; i <= n; i++) {
      path.push(i);
      backtrack(i + 1, path);
      path.pop();
    }
  }
  backtrack(1, []);
  return res;
}
console.log(combine(4, 2));
// [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]```


  点击按钮查看结果
