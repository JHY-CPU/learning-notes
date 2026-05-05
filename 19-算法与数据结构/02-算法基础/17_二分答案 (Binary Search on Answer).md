## Binary Search on Answer


```javascript
在一段范围内二分搜索答案，用判定函数 check(x) 验证可行性。```


```
// 二分答案：求平方根
function mySqrt(x) {
  let l = 0, r = x;
  while (l <= r) {
    const mid = Math.floor((l + r) / 2);
    if (mid * mid <= x && (mid+1)*(mid+1) > x) return mid;
    if (mid * mid < x) l = mid + 1;
    else r = mid - 1;
  }
  return 0;
}
// 分割数组的最大值
function splitArray(nums, k) {
  let l = Math.max(...nums), r = nums.reduce((a,b)=>a+b, 0);
  while (l < r) {
    const mid = Math.floor((l+r)/2);
    let count = 1, sum = 0;
    for (const n of nums) {
      if (sum + n > mid) { count++; sum = n; }
      else sum += n;
    }
    if (count <= k) r = mid;
    else l = mid + 1;
  }
  return l;
}
console.log(mySqrt(8)); // 2
console.log(splitArray([7,2,5,10,8], 2)); // 18```


  点击按钮查看结果
