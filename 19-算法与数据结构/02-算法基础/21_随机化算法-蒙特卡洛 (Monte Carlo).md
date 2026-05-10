# 随机化算法-蒙特卡洛 (Monte Carlo)

## 一、概念定义

### 1.1 蒙特卡洛方法

蒙特卡洛方法是一种通过**大量随机采样**来近似数值结果的算法。

**核心思想：** 当精确计算困难时，用随机实验的统计结果来逼近真值。

### 1.2 两类随机化算法

| 类型 | 特点 | 代表 |
|------|------|------|
| 蒙特卡洛 | 结果可能错误，但运行时间确定 | Miller-Rabin素数测试 |
| 拉斯维加斯 | 结果一定正确，但运行时间随机 | 随机化快速排序 |

---

## 二、经典应用

### 2.1 估算圆周率 PI

用随机点估算单位正方形内切圆的面积比。

```python
import random

def estimate_pi(num_samples):
    inside = 0
    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / num_samples

print(f"PI ≈ {estimate_pi(1000000)}")
# 输出类似: PI ≈ 3.141592
```

**原理：** 单位正方形面积为4，内切圆面积为$\pi$。点落在圆内的概率为 $\pi/4$。

### 2.2 估算定积分

$$I = \int_a^b f(x) dx$$

```python
def monte_carlo_integral(f, a, b, n=100000):
    total = 0
    for _ in range(n):
        x = random.uniform(a, b)
        total += f(x)
    return (b - a) * total / n

# 计算 ∫₀¹ x² dx = 1/3
result = monte_carlo_integral(lambda x: x*x, 0, 1)
print(f"∫x²dx ≈ {result}")  # ≈ 0.333
```

### 2.3 计算不规则图形面积

```python
def area_of_shape(inside_func, bounds, n=100000):
    """inside_func(x,y) 返回点是否在形状内"""
    x_min, x_max, y_min, y_max = bounds
    inside_count = 0
    for _ in range(n):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if inside_func(x, y):
            inside_count += 1
    total_area = (x_max - x_min) * (y_max - y_min)
    return total_area * inside_count / n
```

---

## 三、算法中的蒙特卡洛

### 3.1 Miller-Rabin 素数测试

概率性素数测试，错误率可控制在 $4^{-k}$（$k$轮测试）。

```python
def miller_rabin(n, k=10):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False

    # 将 n-1 写成 d * 2^r
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
```

### 3.2 随机化快速排序

随机选择pivot，避免最坏情况 $O(n^2)$。

```python
import random

def randomized_quicksort(arr, left, right):
    if left >= right: return
    pivot_idx = random.randint(left, right)
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    pivot = arr[right]
    i = left
    for j in range(left, right):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right] = arr[right], arr[i]

    randomized_quicksort(arr, left, i - 1)
    randomized_quicksort(arr, i + 1, right)
```

### 3.3 蓄水池抽样

从数据流中随机抽取k个样本，每个元素被选中的概率相等。

```python
import random

def reservoir_sample(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
```

---

## 四、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 估算PI
double estimatePI(int n) {
    int inside = 0;
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < n; i++) {
        double x = dist(rng), y = dist(rng);
        if (x*x + y*y <= 1.0) inside++;
    }
    return 4.0 * inside / n;
}

// 随机化快速排序
void randomizedQuickSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int pivotIdx = left + rand() % (right - left + 1);
    swap(arr[pivotIdx], arr[right]);
    int pivot = arr[right], i = left;
    for (int j = left; j < right; j++)
        if (arr[j] < pivot) swap(arr[i++], arr[j]);
    swap(arr[i], arr[right]);
    randomizedQuickSort(arr, left, i - 1);
    randomizedQuickSort(arr, i + 1, right);
}
```

---

## 五、复杂度与误差分析

| 算法 | 时间 | 误差 |
|------|------|------|
| 蒙特卡洛PI | $O(n)$ | $O(1/\sqrt{n})$ |
| Miller-Rabin | $O(k \log^2 n)$ | $4^{-k}$ |
| 随机快排 | $O(n \log n)$ 期望 | 无 |
| 蓄水池抽样 | $O(n)$ | 无 |

**误差收敛：** 蒙特卡洛的误差以 $O(1/\sqrt{n})$ 收敛，要提高一位精度需要100倍采样。

---

## 六、面试要点

1. **PI估算** — 最经典的蒙特卡洛入门题
2. **Miller-Rabin** — 理解概率性算法
3. **随机化快排** — 为什么随机化能避免最坏情况
4. **蓄水池抽样** — 数据流场景的经典技巧
