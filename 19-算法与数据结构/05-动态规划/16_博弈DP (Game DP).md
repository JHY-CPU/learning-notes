# 博弈DP (Game DP)

## 1. 概念与定义

博弈DP是基于**博弈论**的动态规划。两名玩家轮流做出最优决策，目标是最大化自己的收益（或最小化对手的收益）。

博弈DP的核心思想：
- 两名玩家都采取**最优策略**
- 每个状态记录**当前玩家**的最优结果
- 转移时考虑**对手也会最优地选择**

常见博弈问题：
- Nim游戏及其变体
- 石子游戏
- 取硬币游戏
- 数字博弈

## 2. 状态定义与转移方程

### 2.1 区间博弈

```
dp[i][j] = 在区间[i,j]上，当前玩家能获得的最大净收益（自己的分-对手的分）
dp[i][j] = max(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1])
dp[i][i] = nums[i]
答案：dp[0][n-1] > 0 表示先手赢
```

### 2.2 Nim游戏

```
经典Nim：有n堆石子，每次从一堆中取若干个
sg(x) = x（每堆石子的SG函数值）
异或和为0则先手必败，否则先手必胜
```

### 2.3 取硬币游戏

```
dp[i] = 有i个硬币时，当前玩家是否能赢
dp[i] = any(not dp[i - coin]) for coin in coins
dp[0] = False（没有硬币可取，当前玩家输）
```

## 3. 算法实现

### 3.1 石子游戏（LeetCode 877）

```python
def stoneGame(piles):
    """区间博弈DP"""
    n = len(piles)
    # dp[i][j] = 在[i,j]区间，当前玩家比对手多拿的分数
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = piles[i]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])

    return dp[0][n - 1] > 0
```

### 3.2 石子游戏II（LeetCode 1140）

```python
def stoneGameII(piles):
    """每次可以取1~2M堆，取完后M = max(M, X)"""
    n = len(piles)
    # 后缀和
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(i, M):
        """从第i堆开始，当前M值，当前玩家能拿到的最大石子数"""
        if i + 2 * M >= n:
            return suffix[i]
        # 取x堆后，对手能拿到的最少石子数
        opponent = min(dp(i + x, max(M, x)) for x in range(1, 2 * M + 1))
        return suffix[i] - opponent

    return dp(0, 1)
```

### 3.3 Nim游戏（LeetCode 292）

```python
def canWinNim(n):
    """经典Nim：每次取1~3个，取到最后一个赢"""
    return n % 4 != 0  # 4的倍数先手必败

# SG函数版Nim
def nimGame(heaps):
    """一般Nim游戏：异或和为0则先手必败"""
    xor_sum = 0
    for h in heaps:
        xor_sum ^= h
    return xor_sum != 0
```

### 3.4 猜数字大小II（LeetCode 375）

```python
def getMoneyAmount(n):
    """最坏情况下猜中数字需要的最少金额"""
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = k + max(dp[i][k - 1] if k > i else 0,
                               dp[k + 1][j] if k < j else 0)
                dp[i][j] = min(dp[i][j], cost)

    return dp[1][n]
```

### 3.5 C++ 实现

```cpp
// 石子游戏
bool stoneGame(vector<int>& piles) {
    int n = piles.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) dp[i][i] = piles[i];
    for (int len = 2; len <= n; len++)
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1]);
        }
    return dp[0][n-1] > 0;
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 石子游戏 | O(n²) | O(n²) |
| 石子游戏II | O(n²M) | O(nM) |
| Nim游戏 | O(n) | O(1) |
| 猜数字II | O(n³) | O(n²) |

## 5. 典型例题

### 例题1：移除石子（LeetCode 1406）

```python
def stoneGameIII(stoneValue):
    """每次取1~3堆"""
    n = len(stoneValue)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(i):
        if i >= n:
            return 0
        result = float('-inf')
        total = 0
        for x in range(3):
            if i + x >= n:
                break
            total += stoneValue[i + x]
            result = max(result, total - dp(i + x + 1))
        return result

    diff = dp(0)
    if diff > 0:
        return "Alice"
    elif diff < 0:
        return "Bob"
    else:
        return "Tie"
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **dp含义混淆**：dp[i]表示当前玩家的结果还是先手的结果
2. **最大化自己的收益 vs 最小化对手的收益**：两者等价但转移不同
3. **边界条件**：最后一步谁赢谁输
4. **SG函数**：理解Sprague-Grundy定理

### 6.2 博弈DP的关键

1. **定义dp状态**：通常 dp[i] 表示当前玩家的最优结果
2. **转移**：当前玩家选最优，对手也会选最优
   ```
   dp[i] = max(选择x的收益 - dp[新状态])  对所有合法选择x
   ```
3. **胜负判断**：dp[初始状态] > 0 先手赢，< 0 先手输

### 6.3 Sprague-Grundy定理

对于多个独立子游戏的组合：
- sg(终态) = 0
- sg(x) = mex{sg(y) : x能转移到y}
- 组合游戏的sg = 各子游戏sg的异或
- sg = 0 则先手必败
