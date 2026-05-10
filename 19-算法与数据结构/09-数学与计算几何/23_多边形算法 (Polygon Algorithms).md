# 多边形算法 (Polygon Algorithms)

## 一、概念定义与原理

### 1.1 多边形分类

- **简单多边形：** 边不自交
- **凸多边形：** 所有内角 $\leq 180°$
- **凹多边形：** 存在内角 $> 180°$

### 1.2 顶点排列

默认按逆时针顺序存储多边形顶点 $p_0, p_1, \ldots, p_{n-1}$。

---

## 二、核心算法

### 2.1 多边形面积（鞋带公式）

$$S = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

其中 $p_n = p_0$。

若顶点按逆时针排列，叉积和为正，面积即为叉积和的一半。

### 2.2 点在多边形内判定

**射线法：** 从待判点向任意方向引一条射线，统计与多边形边的交点数。
- 奇数个交点：在内部
- 偶数个交点：在外部

**绕数法（Winding Number）：** 计算多边形绕待判点的圈数，非零则在内部。

### 2.3 多边形重心

$$G_x = \frac{1}{6S} \sum_{i=0}^{n-1} (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)$$

$$G_y = \frac{1}{6S} \sum_{i=0}^{n-1} (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)$$

---

## 三、代码实现

### 3.1 多边形面积 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    Point(double x=0, double y=0): x(x), y(y) {}
};

double polygon_area(vector<Point>& poly) {
    int n = poly.size();
    double area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return abs(area) / 2;
}
```

### 3.2 点在多边形内 - C++（射线法）

```cpp
// 射线法判断点是否在多边形内
bool point_in_polygon(Point p, vector<Point>& poly) {
    int n = poly.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        // 检查射线 (p 向右) 与边 (poly[j] -> poly[i]) 的交点
        if ((poly[i].y > p.y) != (poly[j].y > p.y)) {
            double x_intersect = poly[j].x + (p.y - poly[j].y) *
                (poly[i].x - poly[j].x) / (poly[i].y - poly[j].y);
            if (p.x < x_intersect) inside = !inside;
        }
    }
    return inside;
}
```

### 3.3 点在多边形内 - C++（绕数法）

```cpp
// 绕数法，返回 0 表示在外部
int winding_number(Point p, vector<Point>& poly) {
    int wn = 0;
    int n = poly.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        if (poly[i].y <= p.y) {
            if (poly[j].y > p.y) {
                double cross = (poly[j].x - poly[i].x) * (p.y - poly[i].y)
                             - (p.x - poly[i].x) * (poly[j].y - poly[i].y);
                if (cross > 0) wn++;
            }
        } else {
            if (poly[j].y <= p.y) {
                double cross = (poly[j].x - poly[i].x) * (p.y - poly[i].y)
                             - (p.x - poly[i].x) * (poly[j].y - poly[i].y);
                if (cross < 0) wn--;
            }
        }
    }
    return wn;
}
```

### 3.4 Python 实现

```python
def polygon_area(poly):
    """鞋带公式求多边形面积"""
    n = len(poly)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2

def point_in_polygon(p, poly):
    """射线法判断点是否在多边形内"""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if (poly[i][1] > p[1]) != (poly[j][1] > p[1]):
            x_intersect = poly[j][0] + (p[1] - poly[j][1]) * \
                (poly[i][0] - poly[j][0]) / (poly[i][1] - poly[j][1])
            if p[0] < x_intersect:
                inside = not inside
        j = i
    return inside

# 测试
square = [(0,0), (2,0), (2,2), (0,2)]
print(polygon_area(square))         # 4.0
print(point_in_polygon((1,1), square))  # True
print(point_in_polygon((3,3), square))  # False
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 面积计算 | $O(n)$ | 鞋带公式 |
| 射线法 | $O(n)$ | 单次查询 |
| 绕数法 | $O(n)$ | 单次查询 |
| 重心计算 | $O(n)$ | |

---

## 五、竞赛与面试应用场景

1. **面积计算：** 不规则多边形面积
2. **包含关系：** 判断点是否在多边形内
3. **多边形并集/交集：** 更高级的计算几何
4. **重心：** 物理建模
