## Palindrome


```javascript
回文串是正读反读都相同的字符串，双指针法 O(n) 判断。```


```
function isPalindrome(s) {
  s = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
  let l = 0, r = s.length - 1;
  while (l < r) {
    if (s[l] !== s[r]) return false;
    l++; r--;
  }
  return true;
}
// 最长回文子串（中心扩展法）
function longestPalindrome(s) {
  let start = 0, end = 0;
  for (let i = 0; i < s.length; i++) {
    const len1 = expand(s, i, i);
    const len2 = expand(s, i, i+1);
    const len = Math.max(len1, len2);
    if (len > end - start) { start = i - Math.floor((len-1)/2); end = i + Math.floor(len/2); }
  }
  return s.substring(start, end+1);
}
function expand(s, l, r) { while (l >= 0 && r < s.length && s[l] === s[r]) { l--; r++; } return r-l-1; }
console.log(isPalindrome("A man, a plan, a canal: Panama")); // true```


  点击按钮查看结果
