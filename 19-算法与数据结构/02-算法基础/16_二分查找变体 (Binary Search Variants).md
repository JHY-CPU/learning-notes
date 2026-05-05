## Binary Search Variants


```javascript
二分查找的变体：找左边界、右边界、第一个大于、最后一个小于等。```


```
// 查找第一个等于 target 的位置
function firstEqual(arr, target) {
  let l = 0, r = arr.length-1, res = -1;
  while (l <= r) {
    const mid = Math.floor((l+r)/2);
    if (arr[mid] >= target) { if (arr[mid] === target) res = mid; r = mid - 1; }
    else l = mid + 1;
  }
  return res;
}
// 查找最后一个等于 target 的位置
function lastEqual(arr, target) {
  let l = 0, r = arr.length-1, res = -1;
  while (l <= r) {
    const mid = Math.floor((l+r)/2);
    if (arr[mid] <= target) { if (arr[mid] === target) res = mid; l = mid + 1; }
    else r = mid - 1;
  }
  return res;
}
console.log(firstEqual([1,2,2,2,3,4], 2)); // 1
console.log(lastEqual([1,2,2,2,3,4], 2)); // 3```


  点击按钮查看结果
