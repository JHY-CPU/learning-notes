## Asymptotic Notation


```javascript
大O表示法描述算法上界，Ω表示下界，Θ表示紧确界。```


```
// 大O表示法常见复杂度排序
// O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)
function compareGrowth(n) {
  return {
    constant: 1,
    logarithmic: Math.log2(n),
    linear: n,
    linearithmic: n * Math.log2(n),
    quadratic: n * n,
    exponential: Math.pow(2, n),
    factorial: n <= 20 ? '太大' : '巨大'
  };
}
console.log(compareGrowth(10));
console.log(compareGrowth(100));```


  点击按钮查看结果
