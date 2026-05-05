## Complexity Basics


```javascript
衡量算法效率的指标：时间复杂度（操作次数）和空间复杂度（内存使用）。```


```
// O(1) - 常数时间
function constantTime(arr) { return arr[0]; }
// O(log n) - 对数时间（二分查找）
function logarithmicTime(n) { let c = 0; for (let i = 1; i < n; i *= 2) c++; return c; }
// O(n) - 线性时间
function linearTime(arr) { let s = 0; for (const x of arr) s += x; return s; }
// O(n log n) - 线性对数
function nlogn(n) { let c = 0; for (let i = 0; i < n; i++) for (let j = 1; j < n; j *= 2) c++; return c; }
// O(n^2) - 平方时间
function quadraticTime(n) { let c = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c++; return c; }
console.log(linearTime([1,2,3,4,5]));```


  点击按钮查看结果
