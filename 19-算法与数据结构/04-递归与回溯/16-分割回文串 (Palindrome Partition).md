## Palindrome Partition


```javascript
将字符串分割成若干回文子串，返回所有可能的分割方案。```


```
function partition(s) {
  const res = [];
  function isPalindrome(str, l, r) {
    while (l < r) if (str[l++] !== str[r--]) return false;
    return true;
  }
  function backtrack(start, path) {
    if (start === s.length) { res.push([...path]); return; }
    for (let i = start; i < s.length; i++) {
      if (isPalindrome(s, start, i)) {
        path.push(s.slice(start, i+1));
        backtrack(i+1, path);
        path.pop();
      }
    }
  }
  backtrack(0, []);
  return res;
}
console.log(partition('aab'));
// [["a","a","b"],["aa","b"]]```


  点击按钮查看结果
