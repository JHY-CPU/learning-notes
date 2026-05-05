## 33-正则表达式匹配 (Regex Match)

  dp[i][j] = match(dp[i-1][j-1]) or star(dp[i][j-2], dp[i-1][j])

  ## 33-正则表达式匹配 (Regex Match)


```javascript
33-正则表达式匹配 (Regex Match) 概念讲解。dp[i][j] = match(dp[i-1][j-1]) or star(dp[i][j-2], dp[i-1][j])```

```
dp[i][j] = match(dp[i-1][j-1]) or star(dp[i][j-2], dp[i-1][j])```
