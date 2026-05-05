## Prefix Sum


```javascript
前缀和将区间查询转化为 O(1) 减法操作，是预计算的核心思想。```


```
// 前缀和基本操作
class PrefixSum {
  constructor(nums) {
    this.prefix = [0];
    for (const n of nums) this.prefix.push(this.prefix[this.prefix.length-1] + n);
  }
  rangeSum(l, r) { return this.prefix[r+1] - this.prefix[l]; }
}
// 二维前缀和
class PrefixSum2D {
  constructor(matrix) {
    const m = matrix.length, n = matrix[0].length;
    this.prefix = Array.from({length:m+1}, () => new Array(n+1).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        this.prefix[i+1][j+1] = matrix[i][j] + this.prefix[i][j+1] + this.prefix[i+1][j] - this.prefix[i][j];
  }
  sumRegion(r1,c1,r2,c2) {
    return this.prefix[r2+1][c2+1] - this.prefix[r1][c2+1] - this.prefix[r2+1][c1] + this.prefix[r1][c1];
  }
}
const ps = new PrefixSum([1,2,3,4,5]);
console.log(ps.rangeSum(1, 3)); // 9```


  点击按钮查看结果
