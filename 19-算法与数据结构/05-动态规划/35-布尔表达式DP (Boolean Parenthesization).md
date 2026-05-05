## 35-布尔表达式DP (Boolean Parenthesization)

  dp[i][j][T/F] = sum(dp[i][k-1] op dp[k+1][j])

  ## 35-布尔表达式DP (Boolean Parenthesization)


```javascript
35-布尔表达式DP (Boolean Parenthesization) 概念讲解。dp[i][j][T/F] = sum(dp[i][k-1] op dp[k+1][j])```

```
dp[i][j][T/F] = sum(dp[i][k-1] op dp[k+1][j])```
