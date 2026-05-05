## Best Worst Average


```javascript
算法在不同输入下的表现不同，需分析最好、最坏、平均情况复杂度。```


```
// 线性查找的最佳/最差/平均情况
function linearSearch(arr, target) {
  for (let i = 0; i < arr.length; i++)
    if (arr[i] === target) return i;
  return -1;
}
// 最好: O(1) - 目标在第一个位置
// 最坏: O(n) - 目标在最后一个或不存在
// 平均: O(n) - 平均检查 n/2 个元素
console.log(linearSearch([1,2,3,4,5], 1)); // 最好
console.log(linearSearch([1,2,3,4,5], 5)); // 最坏```


  点击按钮查看结果
