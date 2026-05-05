## Hash Table Sequence


```javascript
哈希表可以 O(n) 时间找到数组中的最长连续序列。```


```
function longestConsecutive(nums) {
  const set = new Set(nums);
  let longest = 0;
  for (const n of set) {
    if (!set.has(n-1)) { // 序列起点
      let len = 1;
      while (set.has(n + len)) len++;
      longest = Math.max(longest, len);
    }
  }
  return longest;
}
console.log(longestConsecutive([100,4,200,1,3,2])); // 4```


  点击按钮查看结果
