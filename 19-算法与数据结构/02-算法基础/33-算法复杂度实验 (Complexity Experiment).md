## Complexity Experiment


```javascript
通过实际运行观察不同算法的性能差异。```


```
// 对比 O(n) vs O(n²)
function measureTime(fn, n) {
  const start = performance.now();
  fn(n);
  return performance.now() - start;
}
function linear(n) { let s = 0; for (let i = 0; i < n; i++) s += i; return s; }
function quadratic(n) { let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += j; return s; }
// 实验对比
const n = 5000;
console.log(`O(n): ${measureTime(linear, n).toFixed(2)}ms`);
console.log(`O(n²): ${measureTime(quadratic, n).toFixed(2)}ms`);```


  点击按钮查看结果
