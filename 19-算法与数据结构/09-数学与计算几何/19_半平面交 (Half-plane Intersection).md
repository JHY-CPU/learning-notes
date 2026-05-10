# 半平面交 (Half-plane Intersection)

## 一、概念定义与原理

### 1.1 半平面

**半平面**是由一条直线划分平面得到的其中一侧。给定直线 $ax + by + c = 0$，半平面可以表示为 $ax + by + c \leq 0$（或 $\geq 0$）。

### 1.2 半平面交

**半平面交**是多个半平面的公共区域。结果为一个凸多边形（可能为空集或无界）。

### 1.3 应用场景

- 求满足多个线性不等式约束的可行域
- 最小面积/周长外接矩形
- 线性规划（二维）

---

## 二、核心算法

### 2.1 排序 + 双端队列算法

**步骤：**
1. 将每个半平面表示为有向直线（法向量指向保留侧）
2. 按极角排序
3. 用双端队列维护当前交集的边界
4. 每次加入新半平面时，检查队列首尾是否需要弹出
5. 最后检查首尾半平面的交集

**时间复杂度：** $O(n \log n)$

### 2.2 直线与直线求交

给定两条直线，用参数法或代入法求交点。

---

## 三、代码实现

### 3.1 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Point {
    double x, y;
    Point(double x=0, double y=0): x(x), y(y) {}
    Point operator+(const Point& p) const { return {x+p.x, y+p.y}; }
    Point operator-(const Point& p) const { return {x-p.x, y-p.y}; }
    Point operator*(double k) const { return {x*k, y*k}; }
};

double cross(Point a, Point b) { return a.x*b.y - a.y*b.x; }

struct Line {
    Point p, v; // 点 p，方向向量 v，左侧为半平面
    double angle;
    Line() {}
    Line(Point p, Point v): p(p), v(v) { angle = atan2(v.y, v.x); }
    bool operator<(const Line& l) const { return angle < l.angle; }
};

// 判断点 q 在直线 L 左侧（含边界）
bool on_left(Line L, Point q) {
    return cross(L.v, q - L.p) >= 0;
}

// 两直线交点
Point line_intersection(Line a, Line b) {
    Point u = a.p - b.p;
    double t = cross(b.v, u) / cross(a.v, b.v);
    return a.p + a.v * t;
}

// 半平面交，返回凸多边形顶点
vector<Point> halfplane_intersection(vector<Line>& lines) {
    sort(lines.begin(), lines.end());
    deque<Line> dq;
    for (auto& L : lines) {
        while (dq.size() >= 2 && !on_left(L, line_intersection(dq.back(), dq[dq.size()-2])))
            dq.pop_back();
        while (dq.size() >= 2 && !on_left(L, line_intersection(dq.front(), dq[1])))
            dq.pop_front();
        // 处理平行线
        if (!dq.empty() && abs(cross(L.v, dq.back().v)) < 1e-9) {
            if (on_left(dq.back(), L.p)) continue;
            else dq.pop_back();
        }
        dq.push_back(L);
    }
    // 处理首尾
    while (dq.size() >= 3 && !on_left(dq.front(), line_intersection(dq.back(), dq[dq.size()-2])))
        dq.pop_back();
    while (dq.size() >= 3 && !on_left(dq.back(), line_intersection(dq.front(), dq[1])))
        dq.pop_front();

    if (dq.size() < 3) return {}; // 无有界交集
    vector<Point> result;
    for (int i = 0; i < dq.size(); i++) {
        int j = (i + 1) % dq.size();
        result.push_back(line_intersection(dq[i], dq[j]));
    }
    return result;
}
```

### 3.2 Python 实现

```python
def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def line_intersection(p1, v1, p2, v2):
    u = (p1[0]-p2[0], p1[1]-p2[1])
    t = cross(v2, u) / cross(v1, v2)
    return (p1[0]+v1[0]*t, p1[1]+v1[1]*t)

def on_left(p, v, q):
    return cross(v, (q[0]-p[0], q[1]-p[1])) >= 0

def halfplane_intersection(halfplanes):
    """halfplanes: [(p, v)] 表示半平面在直线 p+t*v 的左侧"""
    import math
    halfplanes.sort(key=lambda h: math.atan2(h[1][1], h[1][0]))
    from collections import deque
    dq = deque()
    for p, v in halfplanes:
        while len(dq) >= 2:
            p1, v1 = dq[-2]; p2, v2 = dq[-1]
            inter = line_intersection(p1, v1, p2, v2)
            if not on_left(p, v, inter): dq.pop()
            else: break
        while len(dq) >= 2:
            p1, v1 = dq[0]; p2, v2 = dq[1]
            inter = line_intersection(p1, v1, p2, v2)
            if not on_left(p, v, inter): dq.popleft()
            else: break
        dq.append((p, v))
    return dq
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 排序 | $O(n \log n)$ | 按极角排序 |
| 双端队列扫描 | $O(n)$ | 每条线最多入队出队一次 |
| 总计 | $O(n \log n)$ | |

---

## 五、竞赛与面试应用场景

1. **线性规划（二维）：** 在凸多边形可行域上优化线性目标函数
2. **可行性判断：** 多个约束是否有公共解
3. **最小外接凸多边形：** 逆向使用半平面交
