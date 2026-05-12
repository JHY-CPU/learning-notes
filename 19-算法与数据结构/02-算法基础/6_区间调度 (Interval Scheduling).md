# 07-区间调度 (Interval Scheduling)

区间调度问题是经典的贪心问题：给定若干区间，选出最多的互不重叠区间。

## 算法步骤

1. 按结束时间升序排序
2. 选择第一个区间（结束最早）
3. 依次遍历，若当前开始 >= 上一个选中的结束，则选中

```javascript
function intervalSchedule(intervals) {
  if (!intervals.length) return [];
  intervals.sort((a, b) => a[1] - b[1]);
  const res = [intervals[0]];
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i][0] >= res[res.length - 1][1]) {
      res.push(intervals[i]);
    }
  }
  return res;
}

console.log(intervalSchedule([[1,3],[2,5],[3,6],[5,7],[6,9]]));
// [[1,3],[3,6],[6,9]] - 最多3个不重叠区间
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> intervalSchedule(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });
    vector<vector<int>> res = {intervals[0]};
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] >= res.back()[1]) res.push_back(intervals[i]);
    }
    return res;
}
```

## 变种问题

```javascript
// 1. 会议室数量（最少需要多少间会议室）
function minMeetingRooms(intervals) {
  const starts = intervals.map(i => i[0]).sort((a,b) => a-b);
  const ends = intervals.map(i => i[1]).sort((a,b) => a-b);
  let rooms = 0, endPtr = 0;
  for (let i = 0; i < starts.length; i++) {
    if (starts[i] < ends[endPtr]) rooms++;
    else endPtr++;
  }
  return rooms;
}

// 2. 无重叠区间（最少删除几个使不重叠）
function eraseOverlapIntervals(intervals) {
  intervals.sort((a, b) => a[1] - b[1]);
  let end = intervals[0][1], remove = 0;
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i][0] < end) remove++;
    else end = intervals[i][1];
  }
  return remove;
}

// 3. 用最少的箭引爆气球
function findMinArrowShots(points) {
  points.sort((a, b) => a[1] - b[1]);
  let arrows = 1, end = points[0][1];
  for (let i = 1; i < points.length; i++) {
    if (points[i][0] > end) { arrows++; end = points[i][1]; }
  }
  return arrows;
}
```

## 正确性证明

交换论证法：假设最优解中第一个区间不是结束最早的，将其替换为结束最早的区间不会减少可选区间数。因此贪心解是最优的。

## 复杂度

| 问题 | 时间 | 空间 |
|------|------|------|
| 最多不重叠区间 | O(n log n) | O(1) |
| 最少会议室 | O(n log n) | O(n) |
| 最少删除数 | O(n log n) | O(1) |
| 引爆气球 | O(n log n) | O(1) |

## 何时使用

- 会议安排、课程调度
- 任务调度、资源分配
- 线段覆盖、点覆盖
- 任何"选择最多互不冲突项"的问题

## 常见陷阱

1. **排序键选择**：按结束时间排序，不是开始时间
2. **边界条件**：区间端点相等时的处理（闭区间 vs 开区间）
3. **空输入**：输入为空时的返回值
4. **重复区间**：题目是否允许重合端点
