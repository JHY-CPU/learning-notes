## Interval Scheduling


```javascript
选择最多不重叠区间：按结束时间排序，每次选结束最早的。```


```
function intervalSchedule(intervals) {
  if (!intervals.length) return [];
  intervals.sort((a,b) => a[1] - b[1]); // 按结束时间排序
  const res = [intervals[0]];
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i][0] >= res[res.length-1][1])
      res.push(intervals[i]);
  }
  return res;
}
console.log(intervalSchedule([[1,3],[2,5],[3,6],[5,7],[6,9]]));
// [[1,3],[3,6],[6,9]]```


  点击按钮查看结果
