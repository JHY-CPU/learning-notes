# 19-三分查找 (Ternary Search)

三分查找用于在单峰函数上求极值，将区间三等分后缩小范围。

## 核心思路

```javascript
// 求单峰函数最大值
function ternarySearch(f, l, r, eps = 1e-7) {
  while (r - l > eps) {
    const m1 = l + (r - l) / 3;
    const m2 = r - (r - l) / 3;
    if (f(m1) < f(m2)) l = m1;
    else r = m2;
  }
  return f((l + r) / 2);
}

// 求 f(x) = -(x-3)^2 + 5 的最大值
const f = x => -(x-3)*(x-3) + 5;
console.log(ternarySearch(f, -10, 10)); // 约 5，在 x=3 处取到
```

## C++ 实现

```cpp
#include <cmath>
using namespace std;

double ternarySearch(double (*f)(double), double l, double r, double eps = 1e-7) {
    while (r - l > eps) {
        double m1 = l + (r - l) / 3;
        double m2 = r - (r - l) / 3;
        if (f(m1) < f(m2)) l = m1;
        else r = m2;
    }
    return f((l + r) / 2);
}

// 整数三分查找
int ternarySearchInt(int (*f)(int), int l, int r) {
    while (r - l > 2) {
        int m1 = l + (r - l) / 3;
        int m2 = r - (r - l) / 3;
        if (f(m1) < f(m2)) l = m1;
        else r = m2;
    }
    int best = f(l);
    for (int i = l + 1; i <= r; i++) best = max(best, f(i));
    return best;
}
```

## 整数域三分

```javascript
// 整数版三分
function ternarySearchInt(f, l, r) {
  while (r - l > 2) {
    const m1 = l + Math.floor((r - l) / 3);
    const m2 = r - Math.floor((r - l) / 3);
    if (f(m1) < f(m2)) l = m1;
    else r = m2;
  }
  let best = f(l);
  for (let i = l + 1; i <= r; i++) best = Math.max(best, f(i));
  return best;
}
```

## 三分 vs 二分

| 特性 | 二分 | 三分 |
|------|------|------|
| 数据类型 | 离散有序序列 | 连续单峰函数 |
| 目标 | 查找目标值 | 求极值 |
| 收敛速度 | 每次缩 1/2 | 每次缩 2/3 |
| 精度 | 精确 | 由 eps 控制 |

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 三分查找 | O(log((r-l)/eps)) | O(1) |

## 应用场景

- 几何问题中找最近/最远点
- 单峰函数的最优化
- 机器学习中的学习率调优
- 物理模拟中的最优参数

## 常见陷阱

1. **非单峰函数**：函数必须是严格凸或严格凹的
2. **精度选择**：eps 太小可能导致无限循环
3. **浮点误差**：f(m1) 和 f(m2) 接近时的比较
4. **整数版本**：整数三分要处理步长为 1 的情况
