## Complexity Analysis


```javascript
分析递归算法复杂度的主定理：T(n) = aT(n/b) + f(n)。```


```
// 主定理案例
// T(n) = 2T(n/2) + O(n) → O(n log n) 归并排序
// T(n) = T(n/2) + O(1) → O(log n) 二分查找
// T(n) = 2T(n/2) + O(1) → O(n) 二叉树遍历
// T(n) = T(n-1) + O(1) → O(n)
// T(n) = T(n-1) + O(n) → O(n^2) 选择排序
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n-1) + fibonacci(n-2); // T(n) = T(n-1) + T(n-2) + O(1) → O(2^n)
}
console.log(fibonacci(10)); // 但注意实际n=10较小```


  点击按钮查看结果
