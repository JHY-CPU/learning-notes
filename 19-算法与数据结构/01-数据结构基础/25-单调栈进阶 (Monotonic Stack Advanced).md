## Monotonic Stack Advanced


```javascript
单调栈可以解决接雨水、柱状图最大矩形、每日温度等经典问题。```


```
// 接雨水
function trap(height) {
  let l = 0, r = height.length - 1;
  let lMax = 0, rMax = 0, ans = 0;
  while (l < r) {
    if (height[l] < height[r]) {
      height[l] >= lMax ? lMax = height[l] : ans += lMax - height[l];
      l++;
    } else {
      height[r] >= rMax ? rMax = height[r] : ans += rMax - height[r];
      r--;
    }
  }
  return ans;
}
console.log(trap([0,1,0,2,1,0,1,3,2,1,2,1])); // 6```


  点击按钮查看结果
