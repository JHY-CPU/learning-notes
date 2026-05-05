## Hash Table Counting


```javascript
哈希表是统计频率、词频分析等问题的最佳工具。```


```
// 多数元素
function majorityElement(nums) {
  const count = new Map();
  for (const n of nums) count.set(n, (count.get(n) || 0) + 1);
  for (const [n, c] of count) if (c > nums.length / 2) return n;
  return null;
}
// 找出所有出现次数 > n/3 的元素
function majorityElement2(nums) {
  const count = new Map();
  for (const n of nums) count.set(n, (count.get(n) || 0) + 1);
  const res = [];
  for (const [n, c] of count) if (c > Math.floor(nums.length / 3)) res.push(n);
  return res;
}
console.log(majorityElement([1,2,3,2,2,2,5,4,2])); // 2```


  点击按钮查看结果
