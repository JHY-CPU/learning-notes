# 动态规划专题 (DP Problems)

## 一、概念定义与原理

### 1.1 核心思想

动态规划（DP）将问题分解为重叠子问题，通过**记忆化**避免重复计算。关键要素：

1. **最优子结构：** 最优解包含子问题的最优解
2. **重叠子问题：** 子问题被多次求解
3. **无后效性：** 未来决策不受过去状态的具体路径影响

### 1.2 DP 分类

| 类型 | 特征 | 代表问题 |
|------|------|---------|
| 线性DP | 按顺序递推 | 最长递增子序列 |
| 区间DP | 区间合并 | 石子合并 |
| 背包DP | 选/不选 | 01背包、完全背包 |
| 树形DP | 树上递推 | 树的最大独立集 |
| 状压DP | 状态压缩 | TSP旅行商问题 |
| 数位DP | 按位统计 | 0到n中某数位出现次数 |
| 概率DP | 期望/概率 | 期望步数 |
| DP优化 | 单调队列/斜率优化 | 任务调度 |

---

## 二、核心算法

### 2.1 01背包

$n$ 件物品，每件最多选一次，背包容量 $W$：

$$dp[j] = \max(dp[j], dp[j-w_i] + v_i)$$

### 2.2 最长递增子序列 (LIS)

$$dp[i] = \max(dp[j] + 1) \quad \text{for } j < i, a[j] < a[i]$$

优化：贪心+二分 $O(n \log n)$

### 2.3 最长公共子序列 (LCS)

$$dp[i][j] = \begin{cases} dp[i-1][j-1]+1 & s_1[i]=s_2[j] \\ \max(dp[i-1][j], dp[i][j-1]) & \text{otherwise} \end{cases}$$

---

## 三、代码实现

### 3.1 01背包 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 01 背包（一维优化）
int knapsack_01(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++) {
        for (int j = W; j >= w[i]; j--) { // 逆序！
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    return dp[W];
}

// 完全背包（正序）
int knapsack_complete(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < w.size(); i++) {
        for (int j = w[i]; j <= W; j++) { // 正序！
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    return dp[W];
}
```

### 3.2 LIS O(n log n) - C++

```cpp
int length_of_LIS(vector<int>& nums) {
    vector<int> tails; // tails[i]: 长度为 i+1 的LIS的最小末尾
    for (int x : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else *it = x;
    }
    return tails.size();
}
```

### 3.3 LCS - C++

```cpp
int LCS(string& s1, string& s2) {
    int m = s1.size(), n = s2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
            else dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[m][n];
}
```

### 3.4 区间DP - C++

```cpp
// 石子合并：合并相邻石子堆的最小代价
int stone_merge(vector<int>& stones) {
    int n = stones.size();
    vector<int> prefix(n+1, 0);
    for (int i = 0; i < n; i++) prefix[i+1] = prefix[i] + stones[i];

    vector<vector<int>> dp(n, vector<int>(n, 1e9));
    for (int i = 0; i < n; i++) dp[i][i] = 0;

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            for (int k = i; k < j; k++) {
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + prefix[j+1] - prefix[i]);
            }
        }
    }
    return dp[0][n-1];
}
```

### 3.5 Python 实现

```python
def knapsack_01(w, v, W):
    dp = [0] * (W + 1)
    for wi, vi in zip(w, v):
        for j in range(W, wi - 1, -1):
            dp[j] = max(dp[j], dp[j - wi] + vi)
    return dp[W]

def lis(nums):
    from bisect import bisect_left
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)

def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]+1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

print(knapsack_01([2,3,4,5], [3,4,5,6], 8))  # 10
print(lis([10,9,2,5,3,7,101,18]))              # 4
print(lcs("abcde", "ace"))                      # 3
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 01背包 | $O(nW)$ | $O(W)$ |
| 完全背包 | $O(nW)$ | $O(W)$ |
| LIS $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| LIS $O(n\log n)$ | $O(n \log n)$ | $O(n)$ |
| LCS | $O(mn)$ | $O(mn)$ |
| 区间DP | $O(n^3)$ | $O(n^2)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 300：** 最长递增子序列
2. **LeetCode 1143：** 最长公共子序列
3. **LeetCode 322：** 零钱兑换
4. **LeetCode 518：** 零钱兑换II（完全背包）
5. **LeetCode 1039：** 多边形三角剖分的最低得分（区间DP）
