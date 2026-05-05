## Bitmask DP


```javascript
状态压缩 DP 用整数表示子集状态，常见于旅行商、图着色等问题。```


```
// 状态压缩 DP 示例：分配问题
function assign(jobs, k) {
  const n = jobs.length;
  const dp = new Array(1<= 0; mask--) {
      for (let sub = mask; sub; sub = (sub-1) & mask) {
        dp[mask] = Math.min(dp[mask], Math.max(dp[mask ^ sub], sum[sub]));
      }
    }
  }
  return dp[(1<

  点击按钮查看结果
