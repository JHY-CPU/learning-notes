## Bitwise Applications


```javascript
位运算在算法中的应用包括状态压缩、快速幂、lowbit 操作。```


```
// 快速幂
function fastPow(base, exp) {
  let result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    base *= base;
    exp >>= 1;
  }
  return result;
}
// lowbit 操作
function lowbit(x) { return x & (-x); }
// 统计1的个数
function popcount(x) {
  let c = 0;
  while (x) { c++; x &= x - 1; }
  return c;
}
console.log(fastPow(2, 10)); // 1024
console.log(lowbit(12)); // 4 (12=1100, lowbit=100)
console.log(popcount(11)); // 3 (11=1011)```


  点击按钮查看结果
