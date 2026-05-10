# 计算几何基础 (Computational Geometry)

## 一、概念定义与原理

### 1.1 点与向量

用 $(x, y)$ 表示点或向量。向量运算：
- **加减：** $(x_1, y_1) \pm (x_2, y_2) = (x_1 \pm x_2, y_1 \pm y_2)$
- **数乘：** $k(x, y) = (kx, ky)$
- **点积：** $\vec{a} \cdot \vec{b} = x_1 x_2 + y_1 y_2 = |\vec{a}||\vec{b}|\cos\theta$
- **叉积：** $\vec{a} \times \vec{b} = x_1 y_2 - x_2 y_1 = |\vec{a}||\vec{b}|\sin\theta$

### 1.2 叉积的几何意义

- **正值：** $\vec{b}$ 在 $\vec{a}$ 的逆时针方向
- **负值：** $\vec{b}$ 在 $\vec{a}$ 的顺时针方向
- **零：** 共线

$|\vec{a} \times \vec{b}|$ 等于以 $\vec{a}, \vec{b}$ 为邻边的平行四边形面积。

---

## 二、核心操作

### 2.1 判断点在线段上

点 $P$ 在线段 $AB$ 上 $\Leftrightarrow$ $\vec{AP} \times \vec{AB} = 0$ 且 $P$ 在 $A, B$ 之间。

### 2.2 线段相交

两线段 $AB$ 和 $CD$ 相交 $\Leftrightarrow$：
- $A, B$ 在直线 $CD$ 的两侧
- $C, D$ 在直线 $AB$ 的两侧

（严格相交还需要排除端点情况）

### 2.3 点到直线距离

点 $P$ 到直线 $AB$ 的距离：

$$d = \frac{|\vec{AB} \times \vec{AP}|}{|\vec{AB}|}$$

### 2.4 点到线段距离

分三种情况：投影在线段上、到 $A$ 的距离、到 $B$ 的距离。

---

## 三、代码实现

### 3.1 C++ 几何模板

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    Point(double x=0, double y=0): x(x), y(y) {}
    Point operator+(const Point& p) const { return {x+p.x, y+p.y}; }
    Point operator-(const Point& p) const { return {x-p.x, y-p.y}; }
    Point operator*(double k) const { return {x*k, y*k}; }
    double dot(const Point& p) const { return x*p.x + y*p.y; }
    double cross(const Point& p) const { return x*p.y - y*p.x; }
    double len2() const { return x*x + y*y; }
    double len() const { return sqrt(len2()); }
};

typedef Point Vector;

// 点到直线距离
double point_to_line(Point P, Point A, Point B) {
    Vector v1 = B - A, v2 = P - A;
    return abs(v1.cross(v2)) / v1.len();
}

// 点到线段距离
double point_to_segment(Point P, Point A, Point B) {
    if (A.x == B.x && A.y == B.y) return (P - A).len();
    Vector v1 = B - A, v2 = P - A, v3 = P - B;
    if (v1.dot(v2) < 0) return v2.len();
    if (v1.dot(v3) > 0) return v3.len();
    return point_to_line(P, A, B);
}

// 线段相交（严格）
bool segments_intersect(Point A, Point B, Point C, Point D) {
    Vector ab = B - A, ac = C - A, ad = D - A;
    Vector cd = D - C, ca = A - C, cb = B - C;
    double c1 = ab.cross(ac), c2 = ab.cross(ad);
    double c3 = cd.cross(ca), c4 = cd.cross(cb);
    if (c1 * c2 > 0 || c3 * c4 > 0) return false;
    if (abs(c1) < 1e-9 && abs(c2) < 1e-9) {
        // 共线情况，检查投影重叠
        return !(max(A.x, B.x) < min(C.x, D.x) || max(C.x, D.x) < min(A.x, B.x));
    }
    return true;
}

// 两直线交点
Point line_intersection(Point A, Vector v1, Point B, Vector v2) {
    double t = (B - A).cross(v2) / v1.cross(v2);
    return A + v1 * t;
}
```

### 3.2 Python 实现

```python
import math

def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]

def point_to_segment(P, A, B):
    """点P到线段AB的距离"""
    v1 = (B[0]-A[0], B[1]-A[1])
    v2 = (P[0]-A[0], P[1]-A[1])
    v3 = (P[0]-B[0], P[1]-B[1])
    if dot(v1, v2) < 0:
        return math.sqrt(v2[0]**2 + v2[1]**2)
    if dot(v1, v3) > 0:
        return math.sqrt(v3[0]**2 + v3[1]**2)
    return abs(cross(A, P, B)) / math.sqrt(v1[0]**2 + v1[1]**2)

def segments_intersect(A, B, C, D):
    """判断线段AB和CD是否相交"""
    c1 = cross(A, B, C); c2 = cross(A, B, D)
    c3 = cross(C, D, A); c4 = cross(C, D, B)
    return c1 * c2 <= 0 and c3 * c4 <= 0

print(point_to_segment((1, 1), (0, 0), (2, 0)))  # 1.0
print(segments_intersect((0,0),(2,2),(0,2),(2,0)))  # True
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 叉积/点积 | $O(1)$ | 基础运算 |
| 点到线段距离 | $O(1)$ | |
| 线段相交判断 | $O(1)$ | |
| 两直线交点 | $O(1)$ | |

---

## 五、竞赛与面试应用场景

1. **线段相交：** 图形学、碰撞检测
2. **点到线段距离：** 最近点对问题
3. **多边形面积：** 叉积求和
4. **凸包：** 基于叉积判断转向
5. **三角剖分：** 计算不规则多边形面积
