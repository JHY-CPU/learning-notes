## Master Theorem


```javascript
主定理求解递归式 T(n) = aT(n/b) + f(n) 的渐进复杂度。```


```
// 主定理的三种情况
// T(n) = aT(n/b) + O(n^d)
// 1. d < log_b(a) → O(n^{log_b(a)}) — 归并排序
// 2. d = log_b(a) → O(n^d log n)
// 3. d > log_b(a) → O(n^d) — 一分为二的遍历
//
// 归并排序: T(n) = 2T(n/2) + O(n) → a=2, b=2, d=1 → d=log₂(2)=1 → 情况2 → O(n log n)
// 二分查找: T(n) = T(n/2) + O(1) → a=1, b=2, d=0 → d=log₂(1)=0 → 情况2 → O(log n)
// 二叉树遍历: T(n) = 2T(n/2) + O(1) → a=2, b=2, d=0 → d<1 → O(n)
console.log('主定理帮助快速确定递归算法复杂度');```


  点击按钮查看结果
