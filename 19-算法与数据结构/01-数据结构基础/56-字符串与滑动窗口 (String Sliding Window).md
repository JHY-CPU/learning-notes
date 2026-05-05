## String Sliding Window


```javascript
滑动窗口是字符串子串问题中最重要的技巧，如无重复字符的最长子串。```


```
function lengthOfLongestSubstring(s) {
  const map = new Map();
  let maxLen = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i])) start = Math.max(start, map.get(s[i]) + 1);
    map.set(s[i], i);
    maxLen = Math.max(maxLen, i - start + 1);
  }
  return maxLen;
}
console.log(lengthOfLongestSubstring("abcabcbb")); // 3
console.log(lengthOfLongestSubstring("bbbbb")); // 1```


  点击按钮查看结果
