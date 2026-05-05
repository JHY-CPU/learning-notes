## Hash Table Complexity


```javascript
哈希表的平均时间复杂度为 O(1)，最坏情况下退化到 O(n)。```


```
// 模拟哈希冲突导致性能退化
function benchmark(size) {
  const table = {};
  const start = performance.now();
  for (let i = 0; i < size; i++) table['key_' + i] = i;
  for (let i = 0; i < size; i++) const v = table['key_' + i];
  const end = performance.now();
  return end - start;
}
console.log(benchmark(10000));```


  点击按钮查看结果
