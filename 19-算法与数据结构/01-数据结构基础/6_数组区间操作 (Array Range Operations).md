# 07-数组区间操作 (Array Range Operations)

区间操作关注如何处理数组中的连续子数组片段，包括区间合并、区间交集、区间覆盖等。

## 区间合并

合并所有重叠的区间：

```javascript
// 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
// 输出: [[1,6],[8,10],[15,18]]
function mergeIntervals(intervals) {
  if (intervals.length <= 1) return intervals;
  intervals.sort((a, b) => a[0] - b[0]);
  let result = [intervals[0]];
  for (let i = 1; i < intervals.length; i++) {
    let last = result[result.length - 1];
    if (intervals[i][0] <= last[1]) {
      last[1] = Math.max(last[1], intervals[i][1]);
    } else {
      result.push(intervals[i]);
    }
  }
  return result;
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> result = {intervals[0]};
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] <= result.back()[1]) {
            result.back()[1] = max(result.back()[1], intervals[i][1]);
        } else {
            result.push_back(intervals[i]);
        }
    }
    return result;
}
```

## 区间交集

```javascript
// 求两个区间列表的交集
function intervalIntersection(A, B) {
  let i = 0, j = 0;
  let result = [];
  while (i < A.length && j < B.length) {
    let start = Math.max(A[i][0], B[j][0]);
    let end = Math.min(A[i][1], B[j][1]);
    if (start <= end) result.push([start, end]);
    if (A[i][1] < B[j][1]) i++;
    else j++;
  }
  return result;
}
```

## 区间插入

```javascript
// 插入新区间并合并
function insertInterval(intervals, newInterval) {
  let result = [];
  let i = 0;
  // 添加所有在新区间之前的区间
  while (i < intervals.length && intervals[i][1] < newInterval[0]) {
    result.push(intervals[i++]);
  }
  // 合并重叠区间
  while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
    newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
    newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
    i++;
  }
  result.push(newInterval);
  // 添加剩余区间
  while (i < intervals.length) result.push(intervals[i++]);
  return result;
}
```

## 区间调度（最多不重叠区间）

```javascript
// 贪心：按结束时间排序，每次选最早结束的
function maxNonOverlapping(intervals) {
  intervals.sort((a, b) => a[1] - b[1]);
  let count = 0, end = -Infinity;
  for (let [s, e] of intervals) {
    if (s >= end) {
      count++;
      end = e;
    }
  }
  return count;
}
```

## 差分数组处理区间加减

```javascript
class DiffArray {
  constructor(nums) {
    this.diff = new Array(nums.length);
    this.diff[0] = nums[0];
    for (let i = 1; i < nums.length; i++) {
      this.diff[i] = nums[i] - nums[i - 1];
    }
  }
  // 区间 [l, r] 每个元素加 val
  update(l, r, val) {
    this.diff[l] += val;
    if (r + 1 < this.diff.length) this.diff[r + 1] -= val;
  }
  // 还原结果
  result() {
    let res = new Array(this.diff.length);
    res[0] = this.diff[0];
    for (let i = 1; i < this.diff.length; i++) {
      res[i] = res[i - 1] + this.diff[i];
    }
    return res;
  }
}

// 使用：多次区间加操作
let da = new DiffArray([0, 0, 0, 0, 0]);
da.update(1, 3, 2); // [0,2,2,2,0]
da.update(2, 4, 1); // [0,2,3,3,1]
console.log(da.result()); // [0,2,3,3,1]
```

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 区间合并 | O(n log n) | O(n) |
| 区间交集 | O(m + n) | O(1) |
| 区间插入 | O(n) | O(n) |
| 区间调度 | O(n log n) | O(1) |
| 差分数组更新 | O(1) | O(n) |
