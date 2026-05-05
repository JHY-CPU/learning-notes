## Hash Table Dedup


```javascript
哈希表可以高效地对数组元素去重，时间复杂度 O(n)。```


```
function dedup(arr) {
  return [...new Set(arr)];
}
// 自定义实现
function dedup2(arr) {
  const seen = {};
  const res = [];
  for (const x of arr) {
    if (!seen[x]) { seen[x] = true; res.push(x); }
  }
  return res;
}
console.log(dedup([1,2,2,3,3,3,4])); // [1,2,3,4]```


  点击按钮查看结果
