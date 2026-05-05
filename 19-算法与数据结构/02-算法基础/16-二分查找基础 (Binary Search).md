## Binary Search


```javascript
二分查找在有序数组中 O(log n) 时间定位目标。```


```
function binarySearch(arr, target) {
  let l = 0, r = arr.length - 1;
  while (l <= r) {
    const mid = Math.floor((l + r) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) l = mid + 1;
    else r = mid - 1;
  }
  return -1;
}
// 递归版本
function binarySearchRec(arr, target, l=0, r=arr.length-1) {
  if (l > r) return -1;
  const mid = Math.floor((l + r) / 2);
  if (arr[mid] === target) return mid;
  if (arr[mid] < target) return binarySearchRec(arr, target, mid+1, r);
  return binarySearchRec(arr, target, l, mid-1);
}
console.log(binarySearch([1,3,5,7,9,11], 7)); // 3```


  点击按钮查看结果
