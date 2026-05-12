# 06-贪心算法基础 (Greedy Basics)

贪心算法在每一步选择当前最优解，期望通过局部最优达到全局最优。

## 核心条件

1. **贪心选择性质**：局部最优选择能导致全局最优
2. **最优子结构**：问题的最优解包含子问题的最优解

```javascript
// 零钱兑换（贪心 - 适用于标准货币）
function coinChangeGreedy(coins, amount) {
  coins.sort((a, b) => b - a); // 从大到小
  let count = 0;
  for (const c of coins) {
    while (amount >= c) { amount -= c; count++; }
  }
  return amount === 0 ? count : -1;
}

console.log(coinChangeGreedy([25, 10, 5, 1], 63)); // 6 (25+25+10+1+1+1)
// 注意：[1,3,4] 面额凑 6，贪心得 4+1+1=3个，最优是 3+3=2个
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// 贪心找最大活动数
int activitySelection(vector<pair<int,int>>& activities) {
    sort(activities.begin(), activities.end(),
         [](auto& a, auto& b) { return a.second < b.second; });
    int count = 1, lastEnd = activities[0].second;
    for (int i = 1; i < activities.size(); i++) {
        if (activities[i].first >= lastEnd) {
            count++;
            lastEnd = activities[i].second;
        }
    }
    return count;
}
```

## 贪心 vs 动态规划

| 特性 | 贪心 | 动态规划 |
|------|------|---------|
| 决策方式 | 每步只看当前 | 考虑所有子问题 |
| 正确性 | 需证明 | 状态转移保证 |
| 效率 | 通常 O(n)~O(n log n) | 通常 O(n²)~O(n³) |
| 适用范围 | 更窄 | 更广 |

## 经典贪心问题

```javascript
// 分发糖果（LeetCode 135）
function candy(ratings) {
  const n = ratings.length;
  const candy = new Array(n).fill(1);
  // 从左到右：右边比左边高则糖果+1
  for (let i = 1; i < n; i++) {
    if (ratings[i] > ratings[i-1]) candy[i] = candy[i-1] + 1;
  }
  // 从右到左：左边比右边高则取较大值
  for (let i = n - 2; i >= 0; i--) {
    if (ratings[i] > ratings[i+1]) candy[i] = Math.max(candy[i], candy[i+1] + 1);
  }
  return candy.reduce((a, b) => a + b, 0);
}

// 用最少数量的箭引爆气球（LeetCode 452）
function findMinArrowShots(points) {
  points.sort((a, b) => a[1] - b[1]);
  let arrows = 1, end = points[0][1];
  for (let i = 1; i < points.length; i++) {
    if (points[i][0] > end) { arrows++; end = points[i][1]; }
  }
  return arrows;
}
```

## 贪心正确性证明

用**交换论证法**：
1. 假设存在最优解与贪心解不同
2. 找到第一个不同的选择
3. 证明将最优解中的选择替换为贪心选择不会变差
4. 重复直到最优解与贪心解相同

## 常见陷阱

1. **非标准零钱**：[1,3,4] 凑 6，贪心失败
2. **负数权重**：某些问题有负数时贪心不适用
3. **无法回退**：贪心一旦选择不能撤销
4. **需要证明**：不能仅凭直觉判断贪心正确
