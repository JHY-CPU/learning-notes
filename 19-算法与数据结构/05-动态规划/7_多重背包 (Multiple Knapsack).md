# 多重背包 (Multiple Knapsack)

## 1. 概念与定义

多重背包中**每种物品有给定的数量限制**。
- 有 n 种物品，第 i 种物品重量为 w[i]，价值为 v[i]，数量为 c[i]
- 背包容量为 W
- 目标：不超过背包容量的前提下，使总价值最大

朴素方法是将 c[i] 个相同物品看作 c[i] 个不同物品，转化为0-1背包，但当 c[i] 很大时效率很低。因此需要优化。

主要优化方法：
1. **二进制拆分**：将 c[i] 拆成 log(c[i]) 个物品
2. **单调队列优化**：O(nW) 时间复杂度

## 2. 状态定义与转移方程

### 2.1 朴素DP

```
dp[i][j] = 前i种物品、容量为j时的最大价值
dp[i][j] = max(dp[i-1][j-k*w[i]] + k*v[i]) for k in [0, min(c[i], j/w[i])]
时间复杂度：O(n * W * max(c[i]))
```

### 2.2 二进制拆分优化

将 c[i] 拆分为 1, 2, 4, 8, ..., 2^k, remainder。这些数可以组合出 0~c[i] 的任意值。

```
例如 c[i] = 13 → 拆分为 1, 2, 4, 6
时间复杂度：O(n * W * log(max(c[i])))
```

### 2.3 单调队列优化

按模 w[i] 的余数分组，每组内用单调队列维护窗口最大值，时间复杂度 O(nW)。

## 3. 算法实现

### 3.1 二进制拆分优化（推荐）

```python
def multiple_knapsack_binary(weights, values, counts, capacity):
    # 二进制拆分
    new_weights, new_values = [], []
    for i in range(len(weights)):
        c = counts[i]
        k = 1
        while k <= c:
            new_weights.append(k * weights[i])
            new_values.append(k * values[i])
            c -= k
            k *= 2
        if c > 0:
            new_weights.append(c * weights[i])
            new_values.append(c * values[i])

    # 0-1背包
    dp = [0] * (capacity + 1)
    for i in range(len(new_weights)):
        for j in range(capacity, new_weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - new_weights[i]] + new_values[i])
    return dp[capacity]
```

### 3.2 单调队列优化

```python
from collections import deque

def multiple_knapsack_monotone(weights, values, counts, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        w, v, c = weights[i], values[i], counts[i]
        for r in range(w):
            q = deque()
            for k in range((capacity - r) // w + 1):
                j = r + k * w
                val = dp[j] - k * v
                while q and q[-1][1] <= val:
                    q.pop()
                q.append((k, val))
                while q and q[0][0] < k - c:
                    q.popleft()
                dp[j] = q[0][1] + k * v
    return dp[capacity]
```

### 3.3 C++ 二进制拆分

```cpp
int multipleKnapsack(vector<int>& w, vector<int>& v, vector<int>& c, int W) {
    vector<int> nw, nv;
    for (int i = 0; i < w.size(); i++) {
        int cnt = c[i];
        for (int k = 1; k <= cnt; k *= 2) {
            nw.push_back(k * w[i]);
            nv.push_back(k * v[i]);
            cnt -= k;
        }
        if (cnt > 0) {
            nw.push_back(cnt * w[i]);
            nv.push_back(cnt * v[i]);
        }
    }
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < nw.size(); i++)
        for (int j = W; j >= nw[i]; j--)
            dp[j] = max(dp[j], dp[j - nw[i]] + nv[i]);
    return dp[W];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 朴素枚举 | O(n * W * max(c)) | O(W) |
| 二进制拆分 | O(n * W * log(max(c))) | O(W) |
| 单调队列 | O(n * W) | O(W) |

## 5. 典型例题

### 例题1：多重背包问题

```python
# AcWing 4
N, V = map(int, input().split())
weights, values, counts = [], [], []
for _ in range(N):
    v, w, c = map(int, input().split())
    weights.append(v)
    values.append(w)
    counts.append(c)
# 使用二进制拆分求解
result = multiple_knapsack_binary(weights, values, counts, V)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **二进制拆分遗漏余数**：拆分后 c > 0 的部分需要单独处理
2. **直接转化为0-1背包超时**：当 c[i] 很大时会TLE
3. **单调队列实现错误**：队列维护的索引和值对应关系容易出错

### 6.2 选择优化方法

- c[i] <= 10：朴素方法即可
- c[i] <= 1000：二进制拆分足够
- c[i] 很大且 n*W <= 10^7：单调队列优化
- 竞赛中：二进制拆分最常用

### 6.3 与其他背包的联系

```
c[i] = 1  → 0-1背包
c[i] = ∞  → 完全背包
c[i] = k  → 多重背包
多重背包是0-1背包和完全背包的推广。
```
