## Manacher


```javascript
Manacher 算法 O(n) 时间找出最长回文子串。```


```
function manacher(s) {
  const t = '#' + s.split('').join('#') + '#';
  const p = new Array(t.length).fill(0);
  let center = 0, right = 0;
  for (let i = 0; i < t.length; i++) {
    if (i < right) p[i] = Math.min(right - i, p[2*center - i]);
    while (i-p[i]-1 >= 0 && i+p[i]+1 < t.length && t[i-p[i]-1] === t[i+p[i]+1]) p[i]++;
    if (i + p[i] > right) { center = i; right = i + p[i]; }
  }
  let maxLen = 0, centerIdx = 0;
  for (let i = 0; i < t.length; i++) {
    if (p[i] > maxLen) { maxLen = p[i]; centerIdx = i; }
  }
  const start = (centerIdx - maxLen) / 2;
  return s.slice(start, start + maxLen);
}
console.log(manacher("babad")); // "bab" 或 "aba"
console.log(manacher("cbbd")); // "bb" ```


  点击按钮查看结果
