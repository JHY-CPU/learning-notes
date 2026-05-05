## Math Tools


```javascript
算法分析常用数学工具：对数、指数、级数、模运算、组合数学。```


```
// 常见求和公式
function sumFormulas(n) {
  return {
    sum1: n * (n + 1) / 2,  // 1+2+...+n
    sum2: n * (n + 1) * (2*n+1) / 6, // 1²+2²+...+n²
    sum3: Math.pow(n * (n+1) / 2, 2), // 1³+2³+...+n³
    geometric: Math.pow(2, n+1) - 1, // 1+2+4+...+2^n
  };
}
console.log(sumFormulas(100));```


  点击按钮查看结果
