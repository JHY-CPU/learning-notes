## String Hashing


```javascript
字符串哈希（滚动哈希）将字符串映射为数值，实现 O(1) 子串比较。```


```
// 滚动哈希
function strHash(s, l, r) {
  let h = 0, p = 31, pow = 1;
  for (let i = l; i <= r; i++) {
    h += (s.charCodeAt(i) - 96) * pow;
    pow *= p;
  }
  return h;
}
function rabinKarp(text, pattern) {
  const ph = strHash(pattern, 0, pattern.length-1);
  const res = [];
  for (let i = 0; i <= text.length - pattern.length; i++) {
    if (strHash(text, i, i+pattern.length-1) === ph) {
      let match = true;
      for (let j = 0; j < pattern.length; j++) {
        if (text[i+j] !== pattern[j]) { match = false; break; }
      }
      if (match) res.push(i);
    }
  }
  return res;
}
console.log(rabinKarp("ababcabcabababd", "ababd")); // [10]```


  点击按钮查看结果
