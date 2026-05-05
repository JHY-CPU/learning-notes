## Restore IP


```javascript
给定数字字符串，返回所有有效的IP地址组合。```


```
function restoreIpAddresses(s) {
  const res = [];
  function backtrack(start, path) {
    if (path.length === 4 && start === s.length) { res.push(path.join('.')); return; }
    if (path.length >= 4 || start >= s.length) return;
    for (let len = 1; len <= 3; len++) {
      if (start + len > s.length) break;
      const seg = s.slice(start, start+len);
      if ((seg.length > 1 && seg[0] === '0') || Number(seg) > 255) continue;
      path.push(seg);
      backtrack(start+len, path);
      path.pop();
    }
  }
  backtrack(0, []);
  return res;
}
console.log(restoreIpAddresses('25525511135'));
// ["255.255.11.135","255.255.111.35"]```


  点击按钮查看结果
