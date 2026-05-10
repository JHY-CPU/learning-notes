# 最小圆覆盖 (Minimum Enclosing Circle)

## 一、概念定义与原理

### 1.1 问题定义

给定平面上 $n$ 个点，找到一个半径最小的圆，使得所有点都在圆内或圆上。

### 1.2 基本性质

最小覆盖圆满足以下性质之一：
- 由两个点确定（两点为直径端点）
- 由三个点确定（三点在圆上）

### 1.3 Welzl 算法

随机增量法：按随机顺序逐个加入点，若新点在当前圆外，则更新圆（该点必定在新的边界上）。

期望时间复杂度 $O(n)$。

---

## 二、核心算法

### 2.1 Welzl 算法

**递归过程：**
1. 基础情况：$|P| = 0$ 或 $|R| = 3$，用 $R$ 中的点确定圆
2. 随机选择一个点 $p$
3. 递归求 $P \setminus \{p\}$ 的最小覆盖圆
4. 若 $p$ 不在圆内，将 $p$ 加入 $R$，重新递归

### 2.2 三点确定圆

- 1个点：圆心即该点，半径为0
- 2个点：圆心为中点，半径为距离的一半
- 3个点：求外接圆（三条垂直平分线的交点）

---

## 三、代码实现

### 3.1 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    Point(double x=0, double y=0): x(x), y(y) {}
};

struct Circle {
    Point center;
    double radius;
    Circle(): radius(0) {}
    Circle(Point c, double r): center(c), radius(r) {}
};

double dist(Point a, Point b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}

bool in_circle(Point p, Circle c) {
    return dist(p, c.center) <= c.radius + 1e-9;
}

// 三点确定外接圆
Circle circumcircle(Point a, Point b, Point c) {
    double ax = a.x, ay = a.y, bx = b.x, by = b.y, cx = c.x, cy = c.y;
    double d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by));
    if (abs(d) < 1e-12) {
        // 三点共线，取距离最大的两点
        double d1 = dist(a, b), d2 = dist(b, c), d3 = dist(a, c);
        if (d1 >= d2 && d1 >= d3)
            return Circle(Point((ax+bx)/2, (ay+by)/2), d1/2);
        if (d2 >= d1 && d2 >= d3)
            return Circle(Point((bx+cx)/2, (by+cy)/2), d2/2);
        return Circle(Point((ax+cx)/2, (ay+cy)/2), d3/2);
    }
    double ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d;
    double uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d;
    Point center(ux, uy);
    return Circle(center, dist(center, a));
}

// Welzl 算法
Circle welzl(vector<Point>& points, vector<Point> R, int n) {
    if (n == 0 || R.size() == 3) {
        if (R.empty()) return Circle(Point(0,0), 0);
        if (R.size() == 1) return Circle(R[0], 0);
        if (R.size() == 2) {
            Point mid((R[0].x+R[1].x)/2, (R[0].y+R[1].y)/2);
            return Circle(mid, dist(R[0], R[1])/2);
        }
        return circumcircle(R[0], R[1], R[2]);
    }
    Point p = points[n-1];
    Circle c = welzl(points, R, n-1);
    if (in_circle(p, c)) return c;
    R.push_back(p);
    return welzl(points, R, n-1);
}

Circle minimum_enclosing_circle(vector<Point>& points) {
    mt19937 rng(42);
    shuffle(points.begin(), points.end(), rng);
    return welzl(points, {}, points.size());
}
```

### 3.2 Python 实现

```python
import random, math

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def circumcircle(a, b, c):
    """三点确定外接圆"""
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        # 共线
        d1, d2, d3 = dist(a,b), dist(b,c), dist(a,c)
        if d1 >= d2 and d1 >= d3: return ((ax+bx)/2,(ay+by)/2), d1/2
        if d2 >= d1 and d2 >= d3: return ((bx+cx)/2,(by+cy)/2), d2/2
        return ((ax+cx)/2,(ay+cy)/2), d3/2
    ux = ((ax**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/d
    uy = ((ax**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax-cx)+(cx**2+cy**2)*(bx-ax))/d
    center = (ux, uy)
    return center, dist(center, a)

def minimum_enclosing_circle(points):
    points = list(points)
    random.shuffle(points)
    center, radius = (0,0), 0
    for i, p in enumerate(points):
        if dist(p, center) <= radius + 1e-9: continue
        center, radius = p, 0
        for j in range(i):
            if dist(points[j], center) <= radius + 1e-9: continue
            center = ((p[0]+points[j][0])/2, (p[1]+points[j][1])/2)
            radius = dist(p, points[j]) / 2
            for k in range(j):
                if dist(points[k], center) <= radius + 1e-9: continue
                center, radius = circumcircle(p, points[j], points[k])
    return center, radius

pts = [(0,0), (1,0), (0,1), (1,1)]
c, r = minimum_enclosing_circle(pts)
print(f"Center: {c}, Radius: {r:.4f}")  # (0.5, 0.5), 0.7071
```

---

## 四、复杂度分析

| 算法 | 期望时间复杂度 | 最坏时间复杂度 |
|------|-------------|-------------|
| Welzl（随机增量） | $O(n)$ | $O(n^2)$（极少） |
| 暴力枚举 | $O(n^4)$ | $O(n^4)$ |

---

## 五、竞赛与面试应用场景

1. **最小覆盖圆：** 设施选址、信号覆盖
2. **最大内切圆：** 对偶问题
3. **近似算法：** 在大规模数据上的近似求解
