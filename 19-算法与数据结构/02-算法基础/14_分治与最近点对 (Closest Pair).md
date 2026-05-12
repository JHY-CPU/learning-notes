# 15-分治与最近点对 (Closest Pair)

给定平面上 n 个点，找距离最近的一对点。暴力 O(n^2)，分治 O(n log n)。

## 分治步骤

1. 按 x 坐标排序所有点
2. 以中位数 x 值分为左右两半
3. 递归求左右两半最近距离 d = min(dL, dR)
4. 在中间宽度 2d 的带状区域内检查跨左右的点对
5. 带状区域按 y 排序后每个点只需检查最多 6 个邻居

```javascript
function dist(p1, p2) {
  return Math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2);
}

function closestPairBrute(points) {
  let minD = Infinity;
  for (let i = 0; i < points.length; i++)
    for (let j = i+1; j < points.length; j++)
      minD = Math.min(minD, dist(points[i], points[j]));
  return minD;
}

function closestPair(points) {
  points.sort((a, b) => a[0] - b[0]); // 按 x 排序
  return _closest(points, 0, points.length - 1);
}

function _closest(points, l, r) {
  if (r - l <= 3) return closestPairBrute(points.slice(l, r + 1));

  const mid = (l + r) >> 1;
  const midX = points[mid][0];
  const d = Math.min(_closest(points, l, mid), _closest(points, mid + 1, r));

  // 收集带状区域内的点
  const strip = [];
  for (let i = l; i <= r; i++) {
    if (Math.abs(points[i][0] - midX) < d) strip.push(points[i]);
  }
  strip.sort((a, b) => a[1] - b[1]); // 按 y 排序

  // 每个点检查最多 6 个邻居
  for (let i = 0; i < strip.length; i++) {
    for (let j = i + 1; j < strip.length && strip[j][1] - strip[i][1] < d; j++) {
      const dd = dist(strip[i], strip[j]);
      if (dd < d) d = dd;
    }
  }
  return d;
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
using namespace std;

double dist(pair<double,double>& a, pair<double,double>& b) {
    return hypot(a.first - b.first, a.second - b.second);
}

double closest(vector<pair<double,double>>& pts, int l, int r) {
    if (r - l <= 3) {
        double d = DBL_MAX;
        for (int i = l; i <= r; i++)
            for (int j = i+1; j <= r; j++)
                d = min(d, dist(pts[i], pts[j]));
        return d;
    }
    int m = (l + r) / 2;
    double d = min(closest(pts, l, m), closest(pts, m+1, r));
    double mx = pts[m].first;
    vector<pair<double,double>> strip;
    for (int i = l; i <= r; i++)
        if (abs(pts[i].first - mx) < d) strip.push_back(pts[i]);
    sort(strip.begin(), strip.end(), [](auto& a, auto& b) { return a.second < b.second; });
    for (int i = 0; i < strip.size(); i++)
        for (int j = i+1; j < strip.size() && strip[j].second - strip[i].second < d; j++)
            d = min(d, dist(strip[i], strip[j]));
    return d;
}
```

## 为什么只需检查 6 个邻居

在 d x 2d 的矩形内，任意两点距离 >= d，因此每行最多 2 个点，最多 3 行，共 6 个点。

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 分治 | O(n log n) | O(n) |
| 暴力 | O(n^2) | O(1) |

递推：T(n) = 2T(n/2) + O(n) = O(n log n)。

## 应用

- GPS 导航找最近兴趣点
- 碰撞检测
- 聚类分析最短距离
- 计算几何基础算法
