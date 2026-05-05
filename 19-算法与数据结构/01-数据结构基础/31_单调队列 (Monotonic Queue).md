## Monotonic Queue


```javascript
单调队列中的元素保持单调性，常用于滑动窗口最值问题。```


```
// 滑动窗口最大值
function maxSlidingWindow(nums, k) {
  const q = [], res = [];
  for (let i = 0; i < nums.length; i++) {
    while (q.length && nums[q[q.length-1]] < nums[i]) q.pop();
    q.push(i);
    if (q[0] <= i - k) q.shift();
    if (i >= k - 1) res.push(nums[q[0]]);
  }
  return res;
}
console.log(maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3));
// [3,3,5,5,6,7]```


  点击按钮查看结果
