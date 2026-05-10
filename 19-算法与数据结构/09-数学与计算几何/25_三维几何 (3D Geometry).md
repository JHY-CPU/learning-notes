# 三维几何 (3D Geometry)

## 一、概念定义与原理

### 1.1 三维向量

用 $(x, y, z)$ 表示三维点或向量。

**点积：** $\vec{a} \cdot \vec{b} = x_1 x_2 + y_1 y_2 + z_1 z_2 = |\vec{a}||\vec{b}|\cos\theta$

**叉积：** $\vec{a} \times \vec{b} = (y_1 z_2 - z_1 y_2, z_1 x_2 - x_1 z_2, x_1 y_2 - y_1 x_2)$

叉积结果是一个垂直于 $\vec{a}$ 和 $\vec{b}$ 所在平面的向量。

### 1.2 混合积

$$[\vec{a}, \vec{b}, \vec{c}] = (\vec{a} \times \vec{b}) \cdot \vec{c}$$

其绝对值等于以三向量为棱的平行六面体的体积。

### 1.3 平面方程

一般形式：$ax + by + cz + d = 0$，法向量为 $(a, b, c)$

---

## 二、核心算法

### 2.1 点到平面距离

点 $P(x_0, y_0, z_0)$ 到平面 $ax + by + cz + d = 0$ 的距离：

$$D = \frac{|ax_0 + by_0 + cz_0 + d|}{\sqrt{a^2 + b^2 + c^2}}$$

### 2.2 点到直线距离

$$D = \frac{|\vec{AP} \times \vec{AB}|}{|\vec{AB}|}$$

### 2.3 两直线最短距离

两异面直线 $L_1: A + t\vec{u}$ 和 $L_2: B + s\vec{v}$：

$$d = \frac{|(\vec{AB} \cdot (\vec{u} \times \vec{v}))|}{|\vec{u} \times \vec{v}|}$$

### 2.4 三维凸包

随机增量法，期望 $O(n \log n)$。

---

## 三、代码实现

### 3.1 三维向量模板 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point3 {
    double x, y, z;
    Point3(double x=0, double y=0, double z=0): x(x), y(y), z(z) {}
    Point3 operator+(const Point3& p) const { return {x+p.x, y+p.y, z+p.z}; }
    Point3 operator-(const Point3& p) const { return {x-p.x, y-p.y, z-p.z}; }
    Point3 operator*(double k) const { return {x*k, y*k, z*k}; }
    double dot(const Point3& p) const { return x*p.x + y*p.y + z*p.z; }
    Point3 cross(const Point3& p) const {
        return {y*p.z - z*p.y, z*p.x - x*p.z, x*p.y - y*p.x};
    }
    double len() const { return sqrt(x*x + y*y + z*z); }
};

// 点到平面距离
double point_to_plane(Point3 p, Point3 normal, Point3 on_plane) {
    Point3 v = p - on_plane;
    return abs(v.dot(normal)) / normal.len();
}

// 点到直线距离
double point_to_line3(Point3 P, Point3 A, Point3 B) {
    Point3 v = B - A, w = P - A;
    return v.cross(w).len() / v.len();
}

// 两异面直线最短距离
double line_to_line(Point3 A, Point3 u, Point3 B, Point3 v) {
    Point3 n = u.cross(v);
    if (n.len() < 1e-12) {
        // 平行
        return point_to_line3(A, B, B + v);
    }
    return abs((B - A).dot(n)) / n.len();
}

// 三点确定平面 ax+by+cz+d=0
// 返回 {a, b, c, d}
vector<double> plane_from_3pts(Point3 p1, Point3 p2, Point3 p3) {
    Point3 n = (p2 - p1).cross(p3 - p1);
    double d = -(n.x * p1.x + n.y * p1.y + n.z * p1.z);
    return {n.x, n.y, n.z, d};
}
```

### 3.2 Python 实现

```python
import math

class Point3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __sub__(self, p): return Point3(self.x-p.x, self.y-p.y, self.z-p.z)
    def __add__(self, p): return Point3(self.x+p.x, self.y+p.y, self.z+p.z)
    def __mul__(self, k): return Point3(self.x*k, self.y*k, self.z*k)
    def dot(self, p): return self.x*p.x + self.y*p.y + self.z*p.z
    def cross(self, p):
        return Point3(self.y*p.z-self.z*p.y, self.z*p.x-self.x*p.z, self.x*p.y-self.y*p.x)
    def len(self): return math.sqrt(self.x**2+self.y**2+self.z**2)

def point_to_line3(P, A, B):
    v = B - A; w = P - A
    return v.cross(w).len() / v.len()

def line_to_line(A, u, B, v):
    n = u.cross(v)
    if n.len() < 1e-12: return point_to_line3(A, B, B + v)
    return abs((B - A).dot(n)) / n.len()

# 测试
A = Point3(0,0,0); B = Point3(1,0,0)
P = Point3(0.5, 1, 0)
print(point_to_line3(P, A, B))  # 1.0
```

### 3.3 四面体体积

```cpp
double tetrahedron_volume(Point3 a, Point3 b, Point3 c, Point3 d) {
    return abs((b-a).cross(c-a).dot(d-a)) / 6.0;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 |
|------|-----------|
| 点积/叉积 | $O(1)$ |
| 点到直线距离 | $O(1)$ |
| 两直线距离 | $O(1)$ |
| 三维凸包 | $O(n \log n)$ |

---

## 五、竞赛与面试应用场景

1. **三维空间距离计算：** 物理仿真、机器人路径规划
2. **三维凸包：** 点云处理、形状分析
3. **四面体体积：** 三维多边形面积/体积计算
4. **射线追踪：** 计算机图形学
