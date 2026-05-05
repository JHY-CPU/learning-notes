## Bitwise Problems


```javascript
位运算的常见面试题包括：只出现一次的数字、2的幂、比特位计数等。```


```
// 只出现一次的数字
function singleNumber(nums) {
  return nums.reduce((a,b) => a ^ b, 0);
}
// 比特位计数
function countBits(n) {
  const res = new Array(n+1).fill(0);
  for (let i = 1; i <= n; i++) res[i] = res[i>>1] + (i&1);
  return res;
}
// 2的幂
function isPowerOfTwo(n) { return n > 0 && (n & (n-1)) === 0; }
console.log(singleNumber([4,1,2,1,2])); // 4
console.log(countBits(5)); // [0,1,1,2,1,2]```


  点击按钮查看结果
