## Jump Game


```javascript
给定数组，判断能否从起点跳到终点，贪心维护最远可达位置。```


```
function canJump(nums) {
  let maxReach = 0;
  for (let i = 0; i < nums.length; i++) {
    if (i > maxReach) return false;
    maxReach = Math.max(maxReach, i + nums[i]);
    if (maxReach >= nums.length - 1) return true;
  }
  return true;
}
function minJumps(nums) {
  let jumps = 0, curEnd = 0, farthest = 0;
  for (let i = 0; i < nums.length - 1; i++) {
    farthest = Math.max(farthest, i + nums[i]);
    if (i === curEnd) { jumps++; curEnd = farthest; }
  }
  return jumps;
}
console.log(canJump([2,3,1,1,4])); // true
console.log(canJump([3,2,1,0,4])); // false```


  点击按钮查看结果
