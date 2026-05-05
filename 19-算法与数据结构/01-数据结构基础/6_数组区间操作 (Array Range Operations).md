## 07-数组区间操作 (Array Range Operations)

区间操作关注如何处理数组中的连续子数组片段，包括区间合并、区间交集、区间覆盖等。

## 区间合并

合并所有重叠的区间：

```javascript

// 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
// 输出: [[1,6],[8,10],[15,18]]
function mergeIntervals(intervals) {
  if (intervals.length <= 1) return intervals;

  // 按区间起点排序
  intervals.sort((a, b) => a[0] - b[0]);

  let result = [intervals[0]];

  for (let i = 1; i < intervals.length; i++) {
    let last = result[result.length - 1];
    let current = intervals[i];

    if (current[0] <= last[1]) {
      // 有重叠，合并
      last[1] = Math.max(last[1], current[1]);
    } else {
      // 无重叠，直接添加
      result.push(current);
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

    if (start <= end) {
      result.push([start, end]);
    }

    // 移动结束较早的指针
    if (A[i][1] < B[j][1]) i++;
    else j++;
  }

  return result;
}
```

## 交互演示
