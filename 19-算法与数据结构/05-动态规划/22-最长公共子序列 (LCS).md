## 22-最长公共子序列 (LCS)

  dp[i][j] = match ? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1])

  ## 22-最长公共子序列 (LCS)


```javascript
22-最长公共子序列 (LCS) 概念讲解。dp[i][j] = match ? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1])```

```
dp[i][j] = match ? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1])```
