## 34-戳气球 (Burst Balloons)

  dp[i][j] = max(dp[i][k-1] + dp[k+1][j] + nums[i-1]*nums[k]*nums[j+1])

  ## 34-戳气球 (Burst Balloons)


```javascript
34-戳气球 (Burst Balloons) 概念讲解。dp[i][j] = max(dp[i][k-1] + dp[k+1][j] + nums[i-1]*nums[k]*nums[j+1])```

```
dp[i][j] = max(dp[i][k-1] + dp[k+1][j] + nums[i-1]*nums[k]*nums[j+1])```
