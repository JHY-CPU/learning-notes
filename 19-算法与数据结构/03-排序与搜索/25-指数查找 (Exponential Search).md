## 指数查找 (Exponential Search)

  指数查找先通过指数增长的方式快速缩小查找范围，然后在确定的区间内执行二分查找。


>
    **时间复杂度：**O(log n)。特别适合**无界数据**（如无限流）或目标靠近数组开头的情况。


  ## 算法步骤


    - 从下标 1 开始，以指数增长（i = 1, 2, 4, 8, 16...）检查 arr[i]

    - 如果 arr[i] > target 或超出数组范围，则找到区间 [i/2, min(i, n-1)]

    - 在该区间内执行二分查找



  ## 代码实现


```
function exponentialSearch(arr, target) {
  const n = arr.length;
  if (arr[0] === target) return 0;

  let i = 1;
  while (i < n && arr[i] <= target) {
    i *= 2;
  }

  // 在 [i/2, min(i, n-1)] 区间进行二分查找
  return binarySearch(arr, target, Math.floor(i / 2), Math.min(i, n - 1));
}

function binarySearch(arr, target, left, right) {
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}```

  ## 交互演示
