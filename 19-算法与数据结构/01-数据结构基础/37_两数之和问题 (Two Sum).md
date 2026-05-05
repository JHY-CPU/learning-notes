## Two Sum


```javascript
两数之和是哈希表最经典的应用：在数组中找两个数使其和为目标值。```


```
function twoSum(nums, target) {
  const map = new Map();
  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];
    if (map.has(complement)) return [map.get(complement), i];
    map.set(nums[i], i);
  }
  return [];
}
console.log(twoSum([2,7,11,15], 9)); // [0,1]```


  点击按钮查看结果
