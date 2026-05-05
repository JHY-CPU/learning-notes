## Algorithm Optimization


```javascript
常见的算法优化技巧：剪枝、记忆化、位运算优化、滚动数组、空间换时间。```


```
// 空间换时间：预计算
function precomputePrimes(n) {
  const isPrime = new Array(n+1).fill(true);
  isPrime[0] = isPrime[1] = false;
  for (let i = 2; i*i <= n; i++)
    if (isPrime[i]) for (let j = i*i; j <= n; j += i) isPrime[j] = false;
  return isPrime;
}
// 滚动数组优化空间
function fibSpaceOptimized(n) {
  if (n <= 1) return n;
  let prev2 = 0, prev1 = 1;
  for (let i = 2; i <= n; i++) {
    const cur = prev1 + prev2;
    prev2 = prev1; prev1 = cur;
  }
  return prev1;
}
console.log(fibSpaceOptimized(10)); // 55
console.log('滚动数组将空间从 O(n) 降到 O(1)');```


  点击按钮查看结果
