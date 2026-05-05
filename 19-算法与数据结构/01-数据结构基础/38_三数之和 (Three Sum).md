## Three Sum


```javascript
三数之和使用排序+双指针，或哈希表实现，注意去重处理。```


```
function threeSum(nums) {
  nums.sort((a,b) => a-b);
  const res = [];
  for (let i = 0; i < nums.length-2; i++) {
    if (i > 0 && nums[i] === nums[i-1]) continue;
    let l = i+1, r = nums.length-1;
    while (l < r) {
      const sum = nums[i] + nums[l] + nums[r];
      if (sum === 0) { res.push([nums[i], nums[l], nums[r]]); while (l < r && nums[l] === nums[l+1]) l++; while (l < r && nums[r] === nums[r-1]) r--; l++; r--; }
      else if (sum < 0) l++;
      else r--;
    }
  }
  return res;
}
console.log(threeSum([-1,0,1,2,-1,-4])); // [[-1,-1,2],[-1,0,1]]```


  点击按钮查看结果
