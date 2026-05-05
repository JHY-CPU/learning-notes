## Sliding Window


```javascript
滑动窗口是数组/字符串子区间问题的核心技巧，灵活控制窗口大小。```


```
// 滑动窗口框架
function slidingWindow(s) {
  const window = new Map();
  let l = 0, r = 0;
  while (r < s.length) {
    // 扩大窗口
    const c = s[r++];
    window.set(c, (window.get(c)||0) + 1);
    // 缩小窗口条件
    while (/* 需要缩小 */false) {
      const d = s[l++];
      window.set(d, window.get(d) - 1);
    }
    // 更新答案
  }
}
// 最小覆盖子串
function minWindow(s, t) {
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c)||0)+1);
  let l=0, r=0, valid=0, start=0, len=Infinity;
  const window = new Map();
  while (r < s.length) {
    const c = s[r++];
    if (need.has(c)) {
      window.set(c, (window.get(c)||0)+1);
      if (window.get(c) === need.get(c)) valid++;
    }
    while (valid === need.size) {
      if (r - l < len) { start = l; len = r - l; }
      const d = s[l++];
      if (need.has(d)) {
        if (window.get(d) === need.get(d)) valid--;
        window.set(d, window.get(d)-1);
      }
    }
  }
  return len === Infinity ? '' : s.substr(start, len);
}
console.log(minWindow('ADOBECODEBANC', 'ABC')); // BANC```


  点击按钮查看结果
