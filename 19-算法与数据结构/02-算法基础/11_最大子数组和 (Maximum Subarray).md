## Maximum Subarray


```javascript
Kadane 算法：O(n) 求最大子数组和，分治也可 O(n log n) 解决。```


```
function maxSubArray(nums) {
  let maxSum = nums[0], curSum = nums[0];
  for (let i = 1; i < nums.length; i++) {
    curSum = Math.max(nums[i], curSum + nums[i]);
    maxSum = Math.max(maxSum, curSum);
  }
  return maxSum;
}
// 分治版本
function maxSubArrayDC(nums, l=0, r=nums.length-1) {
  if (l === r) return nums[l];
  const mid = Math.floor((l+r)/2);
  let leftMax = -Infinity, rightMax = -Infinity, sum = 0;
  for (let i = mid; i >= l; i--) { sum += nums[i]; leftMax = Math.max(leftMax, sum); }
  sum = 0;
  for (let i = mid+1; i <= r; i++) { sum += nums[i]; rightMax = Math.max(rightMax, sum); }
  return Math.max(maxSubArrayDC(nums,l,mid), maxSubArrayDC(nums,mid+1,r), leftMax+rightMax);
}
console.log(maxSubArray([-2,1,-3,4,-1,2,1,-5,4])); // 6```


  点击按钮查看结果
