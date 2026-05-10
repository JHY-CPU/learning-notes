# 旋转卡壳 (Rotating Calipers)

## 一、概念定义与原理

### 1.1 核心思想

旋转卡壳是一种在凸多边形上高效求解极值问题的算法。想象用两根平行的"卡尺"夹住凸包，然后旋转卡尺，同时跟踪极值点。

### 1.2 典型应用

1. **凸多边形直径（最远点对）：** $O(n)$
2. **最小外接矩形：** $O(n)$
3. **两凸多边形最近距离：** $O(n + m)$
4. **凸多边形宽度：** $O(n)$

### 1.3 基本原理

在凸多边形上，当一条边旋转时，对踵点（最远点）只会单调移动，不会回退。因此可以用两个指针同步旋转，总时间 $O(n)$。

---

## 二、核心算法

### 2.1 凸多边形直径

给定凸包顶点 $p_0, p_1, \ldots, p_{n-1}$（逆时针）：

1. 对于每条边 $p_i p_{i+1}$，找到离该边最远的点 $p_j$
2. 计算 $|p_i p_j|$ 和 $|p_{i+1} p_j|$，更新最大值
3. 当边旋转到下一个时，$j$ 只需前进（不会后退）

### 2.2 最小外接矩形

对于每条边，找：
- 离该边最远的点（高度）
- 在边的法线方向上最左和最右的点（宽度）

四个极值点确定一个外接矩形。

---

## 三、代码实现

### 3.1 凸多边形直径 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    Point(double x=0, double y=0): x(x), y(y) {}
    Point operator-(const Point& p) const { return {x-p.x, y-p.y}; }
    double dot(const Point& p) const { return x*p.x + y*p.y; }
    double cross(const Point& p) const { return x*p.y - y*p.x; }
    double len2() const { return x*x + y*y; }
};

// 凸多边形直径（最远点对距离的平方）
double convex_diameter(vector<Point>& hull) {
    int n = hull.size();
    if (n <= 1) return 0;
    if (n == 2) return (hull[0] - hull[1]).len2();
    double result = 0;
    int j = 1;
    for (int i = 0; i < n; i++) {
        // 找离边 hull[i]->hull[(i+1)%n] 最远的点
        while ((hull[(i+1)%n] - hull[i]).cross(hull[(j+1)%n] - hull[i]) >
               (hull[(i+1)%n] - hull[i]).cross(hull[j] - hull[i])) {
            j = (j + 1) % n;
        }
        result = max(result, max((hull[i] - hull[j]).len2(),
                                  (hull[(i+1)%n] - hull[j]).len2()));
    }
    return result;
}
```

### 3.2 Python 实现

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def convex_diameter(hull):
    """旋转卡壳求凸多边形直径"""
    n = len(hull)
    if n <= 1: return 0
    if n == 2: return dist2(hull[0], hull[1])
    result = 0
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        while cross(hull[i], hull[ni], hull[(j+1)%n]) > cross(hull[i], hull[ni], hull[j]):
            j = (j + 1) % n
        result = max(result, dist2(hull[i], hull[j]), dist2(hull[ni], hull[j]))
    return result

# 测试：正方形凸包
hull = [(0,0), (1,0), (1,1), (0,1)]
print(convex_diameter(hull))  # 2 (对角线长度平方)
```

### 3.3 两凸包最近距离

```cpp
// 求两个凸多边形之间的最近距离
double convex_distance(vector<Point>& hull1, vector<Point>& hull2) {
    // 找 hull1 最左点和 hull2 最右点
    int a = 0, b = 0;
    int n = hull1.size(), m = hull2.size();
    for (int i = 0; i < n; i++)
        if (hull1[i].x < hull1[a].x) a = i;
    for (int i = 0; i < m; i++)
        if (hull2[i].x > hull2[b].x) b = i;

    double result = 1e18;
    for (int i = 0; i < n; i++) {
        // 旋转卡壳
        Point edge = hull1[(a+1)%n] - hull1[a];
        while (edge.cross(hull2[(b+1)%m] - hull2[a]) < edge.cross(hull2[b] - hull2[a]))
            b = (b + 1) % m;
        result = min(result, point_to_segment_dist(hull2[b], hull1[a], hull1[(a+1)%n]));
        a = (a + 1) % n;
    }
    return result;
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 凸包直径 | $O(n)$ | $O(1)$ |
| 最小外接矩形 | $O(n)$ | $O(1)$ |
| 两凸包距离 | $O(n + m)$ | $O(1)$ |

前提是输入已经是凸包（$O(n \log n)$ 预处理）。

---

## 五、竞赛与面试应用场景

1. **最远点对：** 先凸包，再旋转卡壳
2. **最小矩形覆盖：** 凸包上旋转卡壳
3. **凸多边形面积交：** 配合其他技巧
4. **点集宽度：** 最小距离的平行线对
