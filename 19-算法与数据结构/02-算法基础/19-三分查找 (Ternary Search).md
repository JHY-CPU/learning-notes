## Ternary Search


```javascript
三分查找用于单峰函数（凸函数或凹函数）求极值。```


```
// 三分查找求函数最大值
function ternarySearch(f, l, r, eps=1e-7) {
  while (r - l > eps) {
    const m1 = l + (r - l) / 3;
    const m2 = r - (r - l) / 3;
    if (f(m1) < f(m2)) l = m1;
    else r = m2;
  }
  return f((l + r) / 2);
}
// 抛物线 f(x) = -(x-3)^2 + 5, 最大值在 x=3
const f = (x) => -(x-3)*(x-3) + 5;
console.log(ternarySearch(f, -10, 10)); // 约 5
// 注意：函数必须是单峰（严格的凸或凹函数）```


  点击按钮查看结果
