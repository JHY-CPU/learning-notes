## Hash Table Grouping


```javascript
哈希表可以将具有相同特征的元素分组，如字母异位词分组。```


```
// 字母异位词分组
function groupAnagrams(strs) {
  const map = new Map();
  for (const s of strs) {
    const key = s.split('').sort().join('');
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(s);
  }
  return [...map.values()];
}
console.log(groupAnagrams(["eat","tea","tan","ate","nat","bat"]));
// [["eat","tea","ate"],["tan","nat"],["bat"]]```


  点击按钮查看结果
