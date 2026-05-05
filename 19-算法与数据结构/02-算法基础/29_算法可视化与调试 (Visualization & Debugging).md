## Visualization & Debugging


```javascript
算法调试技巧：打印日志、图形化展示、小规模测试、边界测试。```


```
// 排序过程可视化（打印每次交换）
function debugSort(arr) {
  const steps = [];
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length-1-i; j++) {
      if (arr[j] > arr[j+1]) {
        [arr[j], arr[j+1]] = [arr[j+1], arr[j]];
        steps.push([...arr]);
      }
    }
  }
  return steps;
}
console.log(debugSort([4,2,5,1,3]));
// 打印每一步排序过程```


  点击按钮查看结果
