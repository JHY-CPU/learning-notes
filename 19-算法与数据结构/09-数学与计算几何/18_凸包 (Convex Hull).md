# 凸包 (Convex Hull)

## 一、概念定义与原理

### 1.1 凸集与凸包

**凸集：** 集合中任意两点的连线仍在集合内。

**凸包：** 包含点集中所有点的最小凸多边形。可以理解为用橡皮筋围住所有点后形成的多边形。

### 1.2 叉积基础

对于向量 $\vec{a} = (x_1, y_1)$ 和 $\vec{b} = (x_2, y_2)$：

$$\vec{a} \times \vec{b} = x_1 y_2 - x_2 y_1$$

- $> 0$：$\vec{b}$ 在 $\vec{a}$ 的逆时针方向
- $< 0$：$\vec{b}$ 在 $\vec{a}$ 的顺时针方向
- $= 0$：两向量共线

---

## 二、核心算法

### 2.1 Andrew 算法（单调链）

**步骤：**
1. 将所有点按 $x$ 坐标排序（$x$ 相同按 $y$ 排序）
2. 从左到右扫描，维护下凸壳：每次检查栈顶两个点与当前点的叉积，若非左转则弹栈
3. 从右到左扫描，维护上凸壳
4. 合并得到完整凸包

**优点：** 不需要处理极角排序，实现简单。

### 2.2 Graham 扫描

1. 找到 $y$ 最小的点作为起点（$y$ 相同取 $x$ 最小）
2. 按极角排序其余点
3. 维护一个栈，每次加入新点时弹出非左转的栈顶

---

## 三、代码实现

### 3.1 Andrew 算法 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    bool operator<(const Point& p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
};

double cross(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

vector<Point> convex_hull(vector<Point>& pts) {
    int n = pts.size();
    if (n < 3) return pts;
    sort(pts.begin(), pts.end());
    vector<Point> hull(2 * n);
    int k = 0;
    // 下凸壳
    for (int i = 0; i < n; i++) {
        while (k >= 2 && cross(hull[k-2], hull[k-1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    // 上凸壳
    for (int i = n - 2, t = k + 1; i >= 0; i--) {
        while (k >= t && cross(hull[k-2], hull[k-1], pts[i]) <= 0) k--;
        hull[k++] = pts[i];
    }
    hull.resize(k - 1); // 第一个点重复
    return hull;
}
```

### 3.2 Graham 扫描 - C++

```cpp
vector<Point> graham_scan(vector<Point>& pts) {
    int n = pts.size();
    if (n < 3) return pts;
    // 找最下左的点
    int idx = 0;
    for (int i = 1; i < n; i++) {
        if (pts[i].y < pts[idx].y || (pts[i].y == pts[idx].y && pts[i].x < pts[idx].x))
            idx = i;
    }
    swap(pts[0], pts[idx]);
    Point base = pts[0];
    // 按极角排序
    sort(pts.begin() + 1, pts.end(), [&](const Point& a, const Point& b) {
        double c = cross(base, a, b);
        if (abs(c) < 1e-9) {
            double da = (a.x-base.x)*(a.x-base.x)+(a.y-base.y)*(a.y-base.y);
            double db = (b.x-base.x)*(b.x-base.x)+(b.y-base.y)*(b.y-base.y);
            return da < db;
        }
        return c > 0;
    });
    vector<Point> hull = {pts[0], pts[1]};
    for (int i = 2; i < n; i++) {
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    return hull;
}
```

### 3.3 Python 实现

```python
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1: return points
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

pts = [(0,0), (1,1), (2,0), (1,-1), (0,2), (2,2)]
print(convex_hull(pts))
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Andrew | $O(n \log n)$ | $O(n)$ |
| Graham | $O(n \log n)$ | $O(n)$ |

瓶颈在于排序，扫描过程 $O(n)$。

---

## 五、竞赛与面试应用场景

1. **凸包周长/面积：** 直接计算
2. **最远点对：** 凸包上用旋转卡壳
3. **最小面积矩形：** 凸包上旋转卡壳
4. **点集直径：** 凸包直径
5. **碰撞检测：** 游戏物理引擎
