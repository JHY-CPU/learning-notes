# 图形学基础 - Bezier曲线

## 概述

Bezier曲线是计算机图形学中最重要的参数曲线之一，广泛用于字体设计、路径动画和CAD建模。本节讲解其数学原理和C语言实现。

---

## 1. 线性插值（Lerp）

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double x, y;
} Point2D;

/* 线性插值 */
Point2D lerp(Point2D a, Point2D b, double t) {
    return (Point2D){
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t
    };
}
```

---

## 2. De Casteljau 算法

### 2.1 递归实现

```c
/*
 * De Casteljau 算法
 * 通过反复线性插值计算Bezier曲线上的点
 * 控制点数组 pts，数量 n，参数 t
 */
Point2D de_casteljau(Point2D *pts, int n, double t) {
    if (n == 1) return pts[0];

    Point2D *new_pts = (Point2D *)malloc((n - 1) * sizeof(Point2D));
    for (int i = 0; i < n - 1; i++) {
        new_pts[i] = lerp(pts[i], pts[i + 1], t);
    }

    Point2D result = de_casteljau(new_pts, n - 1, t);
    free(new_pts);
    return result;
}
```

### 2.2 迭代实现

```c
/* De Casteljau 迭代版本（更高效） */
Point2D de_casteljau_iter(Point2D *pts, int n, double t) {
    Point2D *temp = (Point2D *)malloc(n * sizeof(Point2D));
    for (int i = 0; i < n; i++) temp[i] = pts[i];

    for (int r = 1; r < n; r++) {
        for (int i = 0; i < n - r; i++) {
            temp[i] = lerp(temp[i], temp[i + 1], t);
        }
    }

    Point2D result = temp[0];
    free(temp);
    return result;
}
```

---

## 3. 伯恩斯坦多项式

### 3.1 理论基础

n次Bezier曲线定义为：

```
B(t) = sum(i=0..n) C(n,i) * (1-t)^(n-i) * t^i * P_i
```

### 3.2 实现

```c
/* 二项式系数 */
int binomial(int n, int k) {
    if (k > n - k) k = n - k;
    int result = 1;
    for (int i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

/* 伯恩斯坦基函数 */
double bernstein(int n, int i, double t) {
    return binomial(n, i) * pow(1 - t, n - i) * pow(t, i);
}

/* 用伯恩斯坦多项式计算Bezier曲线上的点 */
Point2D bezier_bernstein(Point2D *pts, int n, double t) {
    Point2D result = {0, 0};
    int degree = n - 1;  // 曲线阶数 = 控制点数 - 1

    for (int i = 0; i < n; i++) {
        double b = bernstein(degree, i, t);
        result.x += b * pts[i].x;
        result.y += b * pts[i].y;
    }
    return result;
}
```

---

## 4. 常见Bezier曲线类型

### 4.1 二次Bezier曲线（3个控制点）

```c
/* 二次Bezier曲线: B(t) = (1-t)^2*P0 + 2(1-t)*t*P1 + t^2*P2 */
Point2D quadratic_bezier(Point2D p0, Point2D p1, Point2D p2, double t) {
    double u = 1 - t;
    return (Point2D){
        u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
        u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y
    };
}
```

### 4.2 三次Bezier曲线（4个控制点）

```c
/*
 * 三次Bezier曲线（最常用）
 * B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
 */
Point2D cubic_bezier(Point2D p0, Point2D p1, Point2D p2, Point2D p3, double t) {
    double u = 1 - t;
    double uu = u * u;
    double uuu = uu * u;
    double tt = t * t;
    double ttt = tt * t;

    return (Point2D){
        uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x,
        uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y
    };
}
```

---

## 5. Bezier曲线的导数

```c
/*
 * 三次Bezier曲线的一阶导数
 * B'(t) = 3(1-t)^2*(P1-P0) + 6(1-t)*t*(P2-P1) + 3t^2*(P3-P2)
 */
Point2D cubic_bezier_derivative(Point2D p0, Point2D p1,
                                 Point2D p2, Point2D p3, double t) {
    double u = 1 - t;
    return (Point2D){
        3 * u * u * (p1.x - p0.x) + 6 * u * t * (p2.x - p1.x) + 3 * t * t * (p3.x - p2.x),
        3 * u * u * (p1.y - p0.y) + 6 * u * t * (p2.y - p1.y) + 3 * t * t * (p3.y - p2.y)
    };
}

/* 计算切线角度 */
double cubic_bezier_tangent_angle(Point2D p0, Point2D p1,
                                   Point2D p2, Point2D p3, double t) {
    Point2D d = cubic_bezier_derivative(p0, p1, p2, p3, t);
    return atan2(d.y, d.x);
}
```

---

## 6. Bezier曲线细分

```c
/*
 * De Casteljau细分
 * 将一条Bezier曲线在参数t处分为两段
 * left: 前半段控制点, right: 后半段控制点
 */
void bezier_split(Point2D *pts, int n, double t,
                  Point2D *left, Point2D *right) {
    Point2D *temp = (Point2D *)malloc(n * sizeof(Point2D));
    for (int i = 0; i < n; i++) temp[i] = pts[i];

    left[0] = temp[0];
    right[n - 1] = temp[n - 1];

    for (int r = 1; r < n; r++) {
        left[r] = temp[0];
        right[n - 1 - r] = temp[n - r - 1];

        for (int i = 0; i < n - r; i++) {
            temp[i] = lerp(temp[i], temp[i + 1], t);
        }
    }
    left[n - 1] = right[0] = temp[0];

    free(temp);
}

/*
 * 自适应细分绘制
 * 当曲线足够平坦时用直线近似
 */
double flatness(Point2D *pts, int n) {
    // 检查控制点到端点连线的最大距离
    Point2D p0 = pts[0], pn = pts[n - 1];
    double max_dist = 0;

    for (int i = 1; i < n - 1; i++) {
        // 点到直线距离
        double dx = pn.x - p0.x, dy = pn.y - p0.y;
        double len = sqrt(dx * dx + dy * dy);
        if (len < 1e-10) continue;

        double dist = fabs(dy * pts[i].x - dx * pts[i].y +
                           pn.x * p0.y - pn.y * p0.x) / len;
        if (dist > max_dist) max_dist = dist;
    }
    return max_dist;
}

/* 用画线函数绘制Bezier曲线的回调 */
typedef void (*DrawLineFunc)(int x0, int y0, int x1, int y1);

void bezier_draw_recursive(Point2D *pts, int n, double tolerance,
                           DrawLineFunc draw_line) {
    if (flatness(pts, n) < tolerance) {
        // 足够平坦，画直线
        draw_line((int)round(pts[0].x), (int)round(pts[0].y),
                  (int)round(pts[n-1].x), (int)round(pts[n-1].y));
        return;
    }

    Point2D *left = (Point2D *)malloc(n * sizeof(Point2D));
    Point2D *right = (Point2D *)malloc(n * sizeof(Point2D));
    bezier_split(pts, n, 0.5, left, right);

    bezier_draw_recursive(left, n, tolerance, draw_line);
    bezier_draw_recursive(right, n, tolerance, draw_line);

    free(left);
    free(right);
}
```

---

## 7. 多段Bezier曲线（样条）

```c
/*
 * C0连续: 端点重合
 * C1连续: 端点重合且切线相同
 * 构造C1连续的三次Bezier样条
 */
void build_c1_spline(Point2D *knots, int n_knots,
                     Point2D *controls) {
    // 简单方法：控制点设为端点的三等分点
    for (int i = 0; i < n_knots - 1; i++) {
        controls[i * 4]     = knots[i];      // P0
        controls[i * 4 + 3] = knots[i + 1];  // P3

        // 中间控制点按比例设置
        double dx = knots[i + 1].x - knots[i].x;
        double dy = knots[i + 1].y - knots[i].y;
        controls[i * 4 + 1] = (Point2D){
            knots[i].x + dx / 3.0,
            knots[i].y + dy / 3.0
        };
        controls[i * 4 + 2] = (Point2D){
            knots[i].x + 2 * dx / 3.0,
            knots[i].y + 2 * dy / 3.0
        };
    }
}
```

---

## 8. 测试示例

```c
/* 简易画线函数（控制台） */
#define SCREEN_W 80
#define SCREEN_H 40
char screen[SCREEN_H][SCREEN_W];

void simple_draw_line(int x0, int y0, int x1, int y1) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    while (1) {
        if (x0 >= 0 && x0 < SCREEN_W && y0 >= 0 && y0 < SCREEN_H)
            screen[y0][x0] = '*';
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

int main() {
    memset(screen, ' ', sizeof(screen));

    // 三次Bezier曲线
    Point2D pts[] = {
        {5, 5},     // P0
        {20, 35},   // P1
        {60, 35},   // P2
        {75, 5}     // P3
    };

    // 离散化绘制
    for (double t = 0; t <= 1.0; t += 0.002) {
        Point2D p = cubic_bezier(pts[0], pts[1], pts[2], pts[3], t);
        int x = (int)round(p.x);
        int y = (int)round(p.y);
        if (x >= 0 && x < SCREEN_W && y >= 0 && y < SCREEN_H)
            screen[y][x] = '#';
    }

    // 画控制点
    for (int i = 0; i < 4; i++) {
        int x = (int)pts[i].x, y = (int)pts[i].y;
        if (x >= 0 && x < SCREEN_W && y >= 0 && y < SCREEN_H)
            screen[y][x] = 'O';
    }

    // 显示
    for (int y = SCREEN_H - 1; y >= 0; y--) {
        for (int x = 0; x < SCREEN_W; x++)
            printf("%c", screen[y][x]);
        printf("\n");
    }

    return 0;
}
```

---

## 小结

- De Casteljau算法通过递归线性插值求曲线上的点，直观且数值稳定
- 三次Bezier曲线是实际应用中最常用的类型
- Bezier曲线的导数可用于计算切线方向和曲率
- 自适应细分可以在保证精度的前提下减少绘制开销
- 多段Bezier曲线拼接时需保证连续性条件（C0、C1等）
