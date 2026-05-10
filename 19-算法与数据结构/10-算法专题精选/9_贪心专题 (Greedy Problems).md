# 贪心专题 (Greedy Problems)

## 一、概念定义与原理

### 1.1 贪心算法

**定义：** 在每一步选择中，都采取当前状态下最优的选择，期望通过局部最优达到全局最优。

### 1.2 贪心的适用条件

1. **贪心选择性质：** 每一步的局部最优选择能导致全局最优
2. **最优子结构：** 问题的最优解包含子问题的最优解

### 1.3 贪心 vs DP

| 贪心 | 动态规划 |
|------|---------|
| 每步只考虑当前最优 | 考虑所有子问题 |
| 无回溯 | 可能需要回溯 |
| 证明困难 | 转移方程明确 |
| 效率通常更高 | 保证正确性 |

---

## 二、贪心策略的证明方法

### 2.1 交换论证法

假设存在最优解 $O$ 和贪心解 $G$。证明通过有限次交换，可以将 $O$ 转化为 $G$，且每次交换不增加代价。

### 2.2 调整法

假设贪心解不是最优解，找到一个更好的解，导出矛盾。

### 2.3 区间贪心

- **区间调度：** 按结束时间排序，每次选最早结束的不冲突区间
- **区间覆盖：** 按左端点排序，每次选覆盖最远的区间

---

## 三、代码实现

### 3.1 区间调度（最多不重叠区间）- C++

```cpp
#include <bits/stdc++.h>
using namespace std;

int max_non_overlapping(vector<pair<int,int>>& intervals) {
    sort(intervals.begin(), intervals.end(),
         [](const auto& a, const auto& b) { return a.second < b.second; });
    int count = 0, end = INT_MIN;
    for (auto& [s, e] : intervals) {
        if (s >= end) { count++; end = e; }
    }
    return count;
}
```

### 3.2 区间覆盖 - C++

```cpp
// 用最少的区间覆盖 [start, end]
int min_intervals(vector<pair<int,int>>& intervals, int start, int end) {
    sort(intervals.begin(), intervals.end());
    int count = 0, i = 0, cur_end = start;
    while (cur_end < end) {
        int max_reach = cur_end;
        while (i < intervals.size() && intervals[i].first <= cur_end) {
            max_reach = max(max_reach, intervals[i].second);
            i++;
        }
        if (max_reach == cur_end) return -1; // 无法覆盖
        count++;
        cur_end = max_reach;
    }
    return count;
}
```

### 3.3 分发糖果 - C++

```cpp
// LeetCode 135: 每个孩子至少1个，评分高的比邻居多
int candy(vector<int>& ratings) {
    int n = ratings.size();
    vector<int> candy(n, 1);
    for (int i = 1; i < n; i++) {
        if (ratings[i] > ratings[i-1]) candy[i] = candy[i-1] + 1;
    }
    for (int i = n-2; i >= 0; i--) {
        if (ratings[i] > ratings[i+1]) candy[i] = max(candy[i], candy[i+1] + 1);
    }
    int result = 0;
    for (int c : candy) result += c;
    return result;
}
```

### 3.4 Python 实现

```python
def max_non_overlapping(intervals):
    """最多不重叠区间数"""
    intervals.sort(key=lambda x: x[1])
    count, end = 0, float('-inf')
    for s, e in intervals:
        if s >= end: count += 1; end = e
    return count

def jump_game(nums):
    """LeetCode 55: 跳跃游戏"""
    max_reach = 0
    for i, x in enumerate(nums):
        if i > max_reach: return False
        max_reach = max(max_reach, i + x)
    return True

def min_coins(coins, amount):
    """贪心+DP：零钱兑换（贪心对某些面额有效）"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i: dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

print(max_non_overlapping([(1,3),(2,4),(3,5)]))  # 2
print(jump_game([2,3,1,1,4]))                     # True
```

### 3.5 哈夫曼编码

```cpp
// 哈夫曼编码：贪心构建最优前缀码
int huffman(vector<int>& freq) {
    priority_queue<int, vector<int>, greater<int>> pq(freq.begin(), freq.end());
    int cost = 0;
    while (pq.size() > 1) {
        int a = pq.top(); pq.pop();
        int b = pq.top(); pq.pop();
        cost += a + b;
        pq.push(a + b);
    }
    return cost;
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 区间调度 | $O(n \log n)$ | $O(1)$ |
| 区间覆盖 | $O(n \log n)$ | $O(1)$ |
| 分发糖果 | $O(n)$ | $O(n)$ |
| 哈夫曼编码 | $O(n \log n)$ | $O(n)$ |
| 跳跃游戏 | $O(n)$ | $O(1)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 435：** 无重叠区间
2. **LeetCode 452：** 用最少数量的箭引爆气球
3. **LeetCode 135：** 分发糖果
4. **LeetCode 55：** 跳跃游戏
5. **LeetCode 763：** 划分字母区间
