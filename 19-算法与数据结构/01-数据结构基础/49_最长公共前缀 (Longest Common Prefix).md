## Longest Common Prefix


```javascript
在一组字符串中找出所有字符串共有的最长前缀。```


```
function longestCommonPrefix(strs) {
  if (!strs.length) return '';
  let prefix = strs[0];
  for (let i = 1; i < strs.length; i++) {
    while (strs[i].indexOf(prefix) !== 0) {
      prefix = prefix.slice(0, -1);
      if (!prefix) return '';
    }
  }
  return prefix;
}
console.log(longestCommonPrefix(["flower","flow","flight"])); // fl
console.log(longestCommonPrefix(["dog","racecar","car"])); // ```


  点击按钮查看结果
