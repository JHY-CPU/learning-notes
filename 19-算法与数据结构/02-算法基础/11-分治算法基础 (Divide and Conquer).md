## Divide and Conquer


```javascript
分治：分而治之，将大问题分解为子问题，递归解决后合并结果。```


```
// 分治示例：求数组和
function dcSum(arr) {
  if (arr.length === 0) return 0;
  if (arr.length === 1) return arr[0];
  const mid = Math.floor(arr.length / 2);
  return dcSum(arr.slice(0, mid)) + dcSum(arr.slice(mid));
}
// 分治求最大值
function dcMax(arr) {
  if (arr.length === 1) return arr[0];
  const mid = Math.floor(arr.length / 2);
  return Math.max(dcMax(arr.slice(0, mid)), dcMax(arr.slice(mid)));
}
console.log(dcSum([1,2,3,4,5])); // 15
console.log(dcMax([3,7,2,9,5])); // 9```


  点击按钮查看结果
