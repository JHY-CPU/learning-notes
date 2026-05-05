## String Optimization


```javascript
字符串算法的优化技巧包括预处理、双数组Trie、滚动哈希优化等。```


```
// 双数组 Trie（Double-Array Trie）
// 使用 base 和 check 两个数组实现，查询 O(1)
// 优势：内存紧凑、查询快速
//
// Z 算法（线性时间模式匹配预处理）
function zAlgorithm(s) {
  const n = s.length, z = new Array(n).fill(0);
  let l = 0, r = 0;
  for (let i = 1; i < n; i++) {
    if (i <= r) z[i] = Math.min(r-i+1, z[i-l]);
    while (i+z[i] < n && s[z[i]] === s[i+z[i]]) z[i]++;
    if (i+z[i]-1 > r) { l = i; r = i+z[i]-1; }
  }
  return z;
}
console.log(zAlgorithm("aaabaaaab")); // [0,2,1,0,3,4,2,1,0]
// 应用：字符串匹配时构造 s+$+t，计算 Z 数组```


  点击按钮查看结果
