## String Interview Problems


```javascript
字符串算法在面试中的高频题型。```


```
// 1. 无重复字符的最长字串
function lengthOfLongestSubstring(s) {
  const map = new Map(); let max = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i])) start = Math.max(start, map.get(s[i])+1);
    map.set(s[i], i); max = Math.max(max, i-start+1);
  }
  return max;
}
// 2. 字符串的排列
function checkInclusion(s1, s2) {
  const need = new Map(); for (const c of s1) need.set(c,(need.get(c)||0)+1);
  const win = new Map(); let l=0, r=0, valid=0;
  while (r < s2.length) {
    const c = s2[r++];
    if (need.has(c)) { win.set(c,(win.get(c)||0)+1); if (win.get(c)===need.get(c)) valid++; }
    while (r - l >= s1.length) {
      if (valid === need.size) return true;
      const d = s2[l++];
      if (need.has(d)) { if (win.get(d)===need.get(d)) valid--; win.set(d,win.get(d)-1); }
    }
  }
  return false;
}
console.log(lengthOfLongestSubstring("abcabcbb")); // 3
console.log(checkInclusion("ab", "eidbaooo")); // true```


  点击按钮查看结果
