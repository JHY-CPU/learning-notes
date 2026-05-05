## Sorting Summary


```javascript
常见排序算法的全面对比：时间复杂度、空间复杂度、稳定性。```


```
// 排序算法总览
const sorts = [
  {name:'冒泡', time:'O(n²)', space:'O(1)', stable:'是', inPlace:'是'},
  {name:'选择', time:'O(n²)', space:'O(1)', stable:'否', inPlace:'是'},
  {name:'插入', time:'O(n²)', space:'O(1)', stable:'是', inPlace:'是'},
  {name:'希尔', time:'O(n log n)', space:'O(1)', stable:'否', inPlace:'是'},
  {name:'归并', time:'O(n log n)', space:'O(n)', stable:'是', inPlace:'否'},
  {name:'快排', time:'O(n log n)', space:'O(log n)', stable:'否', inPlace:'是'},
  {name:'堆排', time:'O(n log n)', space:'O(1)', stable:'否', inPlace:'是'},
  {name:'计数', time:'O(n+k)', space:'O(k)', stable:'是', inPlace:'否'},
  {name:'基数', time:'O(d(n+k))', space:'O(n+k)', stable:'是', inPlace:'否'},
  {name:'桶排', time:'O(n+k)', space:'O(n)', stable:'是', inPlace:'否'},
];
console.table(sorts);```


  点击按钮查看结果
