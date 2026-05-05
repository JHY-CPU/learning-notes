## Two Pointers


```javascript
双指针技术包括对撞指针、快慢指针、滑动窗口，是数组问题的核心方法。```


```
// 双指针框架
// 1. 对撞指针：有序数组两数之和、回文判断
// 2. 快慢指针：环检测、中点查找
// 3. 滑动窗口：子串、子数组

// 两数之和 II (已排序)
function twoSumSorted(nums, target) {
  let l = 0, r = nums.length - 1;
  while (l < r) {
    const sum = nums[l] + nums[r];
    if (sum === target) return [l+1, r+1];
    if (sum < target) l++;
    else r--;
  }
  return [-1, -1];
}
// 盛最多水的容器
function maxArea(height) {
  let l = 0, r = height.length - 1, max = 0;
  while (l < r) {
    max = Math.max(max, Math.min(height[l], height[r]) * (r - l));
    if (height[l] < height[r]) l++;
    else r--;
  }
  return max;
}
console.log(twoSumSorted([2,7,11,15], 9)); // [1,2]
console.log(maxArea([1,8,6,2,5,4,8,3,7])); // 49```


  点击按钮查看结果
