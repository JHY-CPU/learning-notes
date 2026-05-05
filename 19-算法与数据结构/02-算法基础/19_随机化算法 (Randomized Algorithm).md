## Randomized Algorithm


```javascript
随机化算法利用随机选择简化问题分析，如随机快速排序、随机化选择。```


```
// 随机化快速排序
function randomizedPartition(arr, l, r) {
  const ri = l + Math.floor(Math.random() * (r - l + 1));
  [arr[ri], arr[r]] = [arr[r], arr[ri]];
  return partition(arr, l, r);
}
function partition(arr, l, r) {
  const pivot = arr[r];
  let i = l - 1;
  for (let j = l; j < r; j++)
    if (arr[j] <= pivot) { i++; [arr[i], arr[j]] = [arr[j], arr[i]]; }
  [arr[i+1], arr[r]] = [arr[r], arr[i+1]];
  return i + 1;
}
// 随机化选择（找第k小）
function randomizedSelect(arr, l, r, k) {
  if (l === r) return arr[l];
  const p = randomizedPartition(arr, l, r);
  const len = p - l + 1;
  if (k === len) return arr[p];
  if (k < len) return randomizedSelect(arr, l, p-1, k);
  return randomizedSelect(arr, p+1, r, k-len);
}
console.log(randomizedSelect([3,2,1,5,4], 0, 4, 2)); // 第2小 = 2```


  点击按钮查看结果
