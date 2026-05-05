## Closest Pair


```javascript
分治法求平面上的最近点对：按x排序，分治递归，在合并区域检查附近点。```


```
// 最近点对的分治思路（伪代码）
function closestPair(points) {
  // 1. 按 x 坐标排序
  // 2. 递归求左右两半的最小距离 d
  // 3. 在中间带状区域检查距离 < d 的点对
  // 4. 取 min(d, 跨区域最小距离)
  // 时间复杂度 O(n log n)
  return '分治算法 O(n log n)';
}
// 暴力法 O(n²)
function bruteForce(points) {
  let min = Infinity;
  for (let i = 0; i < points.length; i++)
    for (let j = i+1; j < points.length; j++)
      min = Math.min(min, Math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2));
  return min;
}
console.log(bruteForce([[0,0],[1,1],[2,2],[3,3]])); // ~1.414```


  点击按钮查看结果
