# 概率DP (Probability DP)

## 1. 概念与定义

概率DP是涉及**概率计算**或**期望值计算**的动态规划问题。在概率DP中：
- **概率DP**：计算某事件发生的概率
- **期望DP**：计算某随机变量的期望值

概率DP的核心公式：
- 全概率公式：P(A) = Σ P(A|Bi) * P(Bi)
- 期望的线性性质：E(aX + bY) = aE(X) + bE(Y)
- 期望DP常见转移：E(i) = Σ P(转移到j) * (E(j) + cost)

## 2. 状态定义与转移方程

### 2.1 概率DP

```
dp[i] = 达到状态i的概率
dp[i] = Σ dp[j] * P(j→i)  对所有能转移到i的状态j
```

### 2.2 期望DP

```
dp[i] = 从状态i出发到达目标的期望步数/代价
dp[i] = Σ P(i→j) * (dp[j] + cost(i,j))
```

### 2.3 掷骰子问题

```
dp[i] = 掷出总和为i的概率
dp[i] = Σ dp[i-k] * (1/6)  for k in [1, 6]
```

### 2.4 从后向前推的期望

```
dp[i] = 从位置i走到终点的期望步数
dp[i] = 1 + Σ (1/6) * dp[i+k]  for k in [1, 6]（如果i+k <= 终点）
```

## 3. 算法实现

### 3.1 掷骰子n次的点数和概率

```python
def diceProbability(n, target):
    """掷n个骰子，点数和为target的概率"""
    # dp[i][j] = 掷i个骰子，点数和为j的概率
    dp = [[0.0] * (6 * n + 1) for _ in range(n + 1)]

    # 初始化：1个骰子
    for j in range(1, 7):
        dp[1][j] = 1.0 / 6

    for i in range(2, n + 1):
        for j in range(i, 6 * i + 1):
            for k in range(1, 7):
                if j - k >= i - 1:  # 前i-1个骰子至少和为i-1
                    dp[i][j] += dp[i - 1][j - k] / 6.0

    return dp[n][target]
```

### 3.2 期望步数：从起点到终点（LeetCode 1227）

```python
def nthPersonGetsNthSeat(n):
    """n个人上飞机，第n个人坐在自己座位上的概率"""
    # 第一个人可能坐在1号座位（概率1/n）或n号座位（概率1/n）或其他座位
    # 答案：第1个人坐1号座位则第n人一定能坐自己座位
    # 第1个人坐n号座位则第n人一定不能坐自己座位
    # 其他情况递归等价
    if n == 1:
        return 1.0
    return 0.5
```

### 3.3 新21点（LeetCode 837）

```python
def new21Game(N, K, W):
    """
    从 [1, W] 中随机抽取，当总和 >= K 时停止
    总和不超过 N 的概率
    """
    if K == 0:
        return 1.0

    dp = [0.0] * (N + W + 1)

    # 当 K <= x <= N 时，概率为1
    for i in range(K, min(N, K + W - 1) + 1):
        dp[i] = 1.0

    # 从后往前推
    # dp[x] = (dp[x+1] + dp[x+2] + ... + dp[x+W]) / W
    # 滑动窗口优化
    window_sum = sum(dp[i] for i in range(K, K + W))
    for i in range(K - 1, -1, -1):
        dp[i] = window_sum / W
        window_sum += dp[i]
        window_sum -= dp[i + W]

    return dp[0]
```

### 3.4 骰子等和概率（LeetCode 1155）

```python
def numRollsToTarget(n, k, target):
    """掷n个k面骰子，点数和为target的方案数"""
    MOD = 10**9 + 7
    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(n):
        new_dp = [0] * (target + 1)
        for j in range(i + 1, min(i * k, target) + 1):
            for face in range(1, min(k, j) + 1):
                if j - face >= 0:
                    new_dp[j] = (new_dp[j] + dp[j - face]) % MOD
        dp = new_dp

    return dp[target]
```

### 3.5 C++ 实现

```cpp
// 新21点
double new21Game(int N, int K, int W) {
    vector<double> dp(N + W + 1, 0.0);
    for (int i = K; i <= N && i < K + W; i++) dp[i] = 1.0;
    double windowSum = 0;
    for (int i = K; i < K + W && i <= N; i++) windowSum += dp[i];
    for (int i = K - 1; i >= 0; i--) {
        dp[i] = windowSum / W;
        windowSum += dp[i] - (i + W <= N + W ? dp[i + W] : 0);
    }
    return dp[0];
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 骰子和概率 | O(n * 6n) | O(n * 6n) |
| 新21点 | O(N+W) | O(N+W) |
| 骰子方案数 | O(n * k * target) | O(target) |

## 5. 典型例题

### 例题1：统计n面骰子方案数（LeetCode 1155）

```python
def numRollsToTarget(n, k, target):
    MOD = 10**9 + 7
    dp = [0] * (target + 1)
    dp[0] = 1
    for _ in range(n):
        new_dp = [0] * (target + 1)
        for j in range(1, target + 1):
            for face in range(1, min(k, j) + 1):
                new_dp[j] = (new_dp[j] + dp[j - face]) % MOD
        dp = new_dp
    return dp[target]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **方向搞错**：概率从前往后推，期望通常从后往前推
2. **浮点精度**：Python浮点精度足够，但注意不要做整数除法
3. **初始化**：dp的初始概率/期望需要正确设置
4. **条件概率**：注意"已发生事件"对概率的影响

### 6.2 期望DP的常见形式

```
E[i] = Σ P(从i转移到j) * (E[j] + cost(i,j))

当有自环（从i可以转移到i自己）时：
E[i] = p * E[i] + Σ_{j≠i} P(i→j) * (E[j] + cost)
E[i] = (Σ_{j≠i} P(i→j) * (E[j] + cost)) / (1 - p)
```

### 6.3 优化技巧

1. **滑动窗口**：当转移涉及连续区间和时，用滑动窗口优化
2. **前缀和优化**：当转移涉及前缀和时
3. **空间优化**：用滚动数组减少空间
