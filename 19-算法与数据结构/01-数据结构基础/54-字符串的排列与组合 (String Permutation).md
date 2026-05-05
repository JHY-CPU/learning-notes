## String Permutation


```javascript
字符串的全排列和组合是递归回溯的经典应用。```


```
function permute(s) {
  const res = [];
  function backtrack(path, used) {
    if (path.length === s.length) { res.push(path); return; }
    for (let i = 0; i < s.length; i++) {
      if (used[i]) continue;
      if (i > 0 && s[i] === s[i-1] && !used[i-1]) continue;
      used[i] = true;
      backtrack(path + s[i], used);
      used[i] = false;
    }
  }
  const arr = s.split('').sort();
  backtrack('', new Array(s.length).fill(false));
  return res;
}
console.log(permute("aab")); // ["aab","aba","baa"]```


  点击按钮查看结果
