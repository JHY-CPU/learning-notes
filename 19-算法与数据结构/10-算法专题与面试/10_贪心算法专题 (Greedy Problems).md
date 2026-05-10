# 贪心算法专题 (Greedy Problems)

## 一、概念定义与原理

### 1.1 贪心思想

贪心算法在每一步选择中都采取**当前最优**的选择，期望通过局部最优达到全局最优。

**适用条件：**
1. **贪心选择性质：** 局部最优解能构成全局最优解
2. **最优子结构：** 问题的最优解包含子问题的最优解

**与DP的区别：**
- 贪心：每步只做一次决策，不可回退
- DP：考虑所有子问题，通过状态转移得出最优

### 1.2 证明贪心正确性

常用方法：
- **反证法：** 假设存在更优解，推导矛盾
- **交换论证法：** 证明任意解可通过交换调整为贪心解且不变差
- **归纳法：** 逐步证明每步选择保持最优性

---

## 二、经典题目详解

### 2.1 区间调度问题 (LeetCode 435)

**问题：** 给定区间集合，移除最少的区间使剩余区间互不重叠。

**贪心策略：** 按结束时间排序，每次选结束最早的。

```python
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count, end = 0, float('-inf')
    for start, finish in intervals:
        if start >= end:
            end = finish
        else:
            count += 1
    return count
```

### 2.2 跳跃游戏 (LeetCode 55)

**问题：** 判断能否从起点跳到终点。

```python
def can_jump(nums):
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach: return False
        max_reach = max(max_reach, i + jump)
    return True
```

### 2.3 跳跃游戏II (LeetCode 45)

**问题：** 最少跳跃次数。

```python
def jump(nums):
    jumps, cur_end, farthest = 0, 0, 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = farthest
    return jumps
```

### 2.4 分发糖果 (LeetCode 135)

```python
def candy(ratings):
    n = len(ratings)
    candies = [1] * n
    # 从左到右
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    # 从右到左
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)
```

### 2.5 加油站 (LeetCode 134)

```python
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    start, tank = 0, 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

### 2.6 用最少数量的箭引爆气球 (LeetCode 452)

```python
def find_min_arrow_shots(points):
    points.sort(key=lambda x: x[1])
    arrows, end = 0, float('-inf')
    for start, finish in points:
        if start > end:
            arrows += 1
            end = finish
    return arrows
```

### 2.7 任务调度器 (LeetCode 621)

```python
def least_interval(tasks, n):
    from collections import Counter
    freq = Counter(tasks)
    max_f = max(freq.values())
    max_count = sum(1 for v in freq.values() if v == max_f)
    return max(len(tasks), (max_f - 1) * (n + 1) + max_count)
```

---

## 三、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 跳跃游戏
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }
    return true;
}

// 区间调度
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });
    int count = 0, end = INT_MIN;
    for (auto& inv : intervals) {
        if (inv[0] >= end) end = inv[1];
        else count++;
    }
    return count;
}

// 分发糖果
int candy(vector<int>& ratings) {
    int n = ratings.size();
    vector<int> candies(n, 1);
    for (int i = 1; i < n; i++)
        if (ratings[i] > ratings[i-1])
            candies[i] = candies[i-1] + 1;
    for (int i = n-2; i >= 0; i--)
        if (ratings[i] > ratings[i+1])
            candies[i] = max(candies[i], candies[i+1] + 1);
    return accumulate(candies.begin(), candies.end(), 0);
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 区间调度 | $O(n \log n)$ | $O(1)$ |
| 跳跃游戏 | $O(n)$ | $O(1)$ |
| 分发糖果 | $O(n)$ | $O(n)$ |
| 加油站 | $O(n)$ | $O(1)$ |
| 任务调度器 | $O(n)$ | $O(26)$ |

---

## 五、面试高频题

1. **LeetCode 55：** 跳跃游戏
2. **LeetCode 45：** 跳跃游戏II
3. **LeetCode 435：** 无重叠区间
4. **LeetCode 135：** 分发糖果
5. **LeetCode 134：** 加油站
6. **LeetCode 452：** 用最少数量的箭引爆气球
7. **LeetCode 621：** 任务调度器
8. **LeetCode 763：** 划分字母区间
9. **LeetCode 406：** 根据身高重建队列
10. **LeetCode 122：** 买卖股票的最佳时机II
