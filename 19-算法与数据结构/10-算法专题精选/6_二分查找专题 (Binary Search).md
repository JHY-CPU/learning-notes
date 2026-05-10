# 二分查找专题 (Binary Search)

## 一、概念定义与原理

### 1.1 基本思想

在**有序**数据中查找目标值，每次将搜索范围缩小一半。时间复杂度 $O(\log n)$。

### 1.2 三种变体

1. **查找等于 target 的位置**
2. **查找第一个 $\geq$ target 的位置（lower_bound）**
3. **查找第一个 $>$ target 的位置（upper_bound）**

### 1.3 二分答案

当直接求解困难，但**验证答案是否可行**容易时，可以二分搜索答案空间。

---

## 二、核心算法

### 2.1 基本二分

```cpp
int binary_search(vector<int>& a, int target) {
    int l = 0, r = a.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] == target) return m;
        else if (a[m] < target) l = m + 1;
        else r = m - 1;
    }
    return -1;
}
```

### 2.2 lower_bound / upper_bound

```cpp
// 第一个 >= target 的位置
int lower_bound(vector<int>& a, int target) {
    int l = 0, r = a.size();
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] < target) l = m + 1;
        else r = m;
    }
    return l;
}

// 第一个 > target 的位置
int upper_bound(vector<int>& a, int target) {
    int l = 0, r = a.size();
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] <= target) l = m + 1;
        else r = m;
    }
    return l;
}
```

---

## 三、代码实现

### 3.1 二分答案模板 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 二分答案：求满足条件的最小值
long long binary_search_min(long long lo, long long hi, function<bool(long long)> check) {
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (check(mid)) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// 二分答案：求满足条件的最大值
long long binary_search_max(long long lo, long long hi, function<bool(long long)> check) {
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (check(mid)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}
```

### 3.2 浮点二分

```cpp
// 求平方根
double sqrt_custom(double x) {
    double lo = 0, hi = max(1.0, x);
    for (int i = 0; i < 200; i++) {
        double mid = (lo + hi) / 2;
        if (mid * mid < x) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}
```

### 3.3 经典例题：分割数组的最大值

```cpp
// 将数组分成 m 段，使得最大段和最小
bool can_split(vector<int>& nums, int m, long long max_sum) {
    int count = 1;
    long long current = 0;
    for (int x : nums) {
        if (current + x > max_sum) { count++; current = 0; }
        current += x;
    }
    return count <= m;
}

int split_array(vector<int>& nums, int m) {
    long long lo = *max_element(nums.begin(), nums.end());
    long long hi = 0;
    for (int x : nums) hi += x;
    return binary_search_min(lo, hi, [&](long long mid) {
        return can_split(nums, m, mid);
    });
}
```

### 3.4 Python 实现

```python
import bisect

def lower_bound(a, target):
    """第一个 >= target 的位置"""
    return bisect.bisect_left(a, target)

def upper_bound(a, target):
    """第一个 > target 的位置"""
    return bisect.bisect_right(a, target)

def binary_search_min(lo, hi, check):
    """二分求满足条件的最小值"""
    while lo < hi:
        mid = (lo + hi) // 2
        if check(mid): hi = mid
        else: lo = mid + 1
    return lo

def sqrt_custom(x):
    lo, hi = 0, max(1.0, x)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid * mid < x: lo = mid
        else: hi = mid
    return (lo + hi) / 2

print(lower_bound([1,2,4,4,5], 4))  # 2
print(upper_bound([1,2,4,4,5], 4))  # 4
print(sqrt_custom(2))               # 1.41421356...
```

### 3.5 旋转排序数组中搜索

```cpp
int search_rotated(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (nums[m] == target) return m;
        if (nums[l] <= nums[m]) { // 左半有序
            if (nums[l] <= target && target < nums[m]) r = m - 1;
            else l = m + 1;
        } else { // 右半有序
            if (nums[m] < target && target <= nums[r]) l = m + 1;
            else r = m - 1;
        }
    }
    return -1;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 基本二分 | $O(\log n)$ | $O(1)$ |
| 二分答案 | $O(\log(\text{范围}) \times \text{check})$ | $O(1)$ |
| 浮点二分 | $O(\log(\text{精度}^{-1}))$ | $O(1)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 34：** 排序数组中查找元素的第一个和最后一个位置
2. **LeetCode 33：** 搜索旋转排序数组
3. **LeetCode 410：** 分割数组的最大值（二分答案）
4. **LeetCode 875：** 爱吃香蕉的珂珂
5. **最小化最大值/最大化最小值：** 经典二分答案模型
