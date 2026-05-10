# 分治专题 (Divide and Conquer)

## 一、概念定义与原理

### 1.1 分治思想

将问题分解为若干个规模较小的子问题，递归求解子问题，然后合并结果。

**三步骤：**
1. **分解 (Divide)：** 将问题分成子问题
2. **解决 (Conquer)：** 递归求解子问题
3. **合并 (Combine)：** 将子问题的解合并为原问题的解

### 1.2 分治 vs DP

| 分治 | 动态规划 |
|------|---------|
| 子问题不重叠 | 子问题重叠 |
| 递归求解 | 自底向上或记忆化 |
| 合并步骤是关键 | 转移方程是关键 |

---

## 二、经典问题

### 2.1 归并排序

分解：将数组一分为二；解决：递归排序；合并：合并两个有序数组。

### 2.2 快速幂

$a^n = (a^{n/2})^2$（$n$ 为偶数），$a^n = a \cdot (a^{(n-1)/2})^2$（$n$ 为奇数）。

### 2.3 最近点对

分解：按 $x$ 坐标一分为二；解决：递归求解；合并：处理跨中线的点对。

### 2.4 最大子数组和

分解：中点左边、右边、跨越中点三种情况取最大。

---

## 三、代码实现

### 3.1 归并排序 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

void merge_sort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    merge_sort(a, l, m);
    merge_sort(a, m+1, r);
    vector<int> temp;
    int i = l, j = m + 1;
    while (i <= m && j <= r) {
        if (a[i] <= a[j]) temp.push_back(a[i++]);
        else temp.push_back(a[j++]);
    }
    while (i <= m) temp.push_back(a[i++]);
    while (j <= r) temp.push_back(a[j++]);
    copy(temp.begin(), temp.end(), a.begin() + l);
}
```

### 3.2 快速幂 - C++

```cpp
long long power(long long a, long long n, long long mod) {
    if (n == 0) return 1;
    long long half = power(a, n / 2, mod);
    long long result = half * half % mod;
    if (n & 1) result = result * a % mod;
    return result;
}
```

### 3.3 最大子数组和（分治法）- C++

```cpp
int max_crossing(vector<int>& a, int l, int m, int r) {
    int left_sum = INT_MIN, sum = 0;
    for (int i = m; i >= l; i--) { sum += a[i]; left_sum = max(left_sum, sum); }
    int right_sum = INT_MIN; sum = 0;
    for (int i = m+1; i <= r; i++) { sum += a[i]; right_sum = max(right_sum, sum); }
    return left_sum + right_sum;
}

int max_subarray(vector<int>& a, int l, int r) {
    if (l == r) return a[l];
    int m = l + (r - l) / 2;
    return max({max_subarray(a, l, m),
                max_subarray(a, m+1, r),
                max_crossing(a, l, m, r)});
}
```

### 3.4 Python 实现

```python
def merge_sort(a):
    if len(a) <= 1: return a
    mid = len(a) // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    result = []; i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    result += left[i:] + right[j:]
    return result

def power(a, n, mod):
    if n == 0: return 1
    half = power(a, n // 2, mod)
    result = half * half % mod
    if n & 1: result = result * a % mod
    return result

def max_subarray(a):
    if len(a) == 1: return a[0]
    mid = len(a) // 2
    left_max = max_subarray(a[:mid])
    right_max = max_subarray(a[mid:])
    left_sum = right_sum = float('-inf')
    s = 0
    for i in range(mid-1, -1, -1): s += a[i]; left_sum = max(left_sum, s)
    s = 0
    for i in range(mid, len(a)): s += a[i]; right_sum = max(right_sum, s)
    return max(left_max, right_max, left_sum + right_sum)

print(merge_sort([5,2,3,1]))      # [1,2,3,5]
print(power(2, 10, 10**9+7))      # 1024
print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
```

### 3.5 最近点对

```cpp
double closest_pair(vector<pair<double,double>>& pts, int l, int r) {
    if (r - l <= 3) {
        double d = 1e18;
        for (int i = l; i <= r; i++)
            for (int j = i+1; j <= r; j++)
                d = min(d, hypot(pts[i].first-pts[j].first, pts[i].second-pts[j].second));
        return d;
    }
    int m = (l + r) / 2;
    double d = min(closest_pair(pts, l, m), closest_pair(pts, m+1, r));
    // 合并：检查中线两侧距离 < d 的点
    vector<pair<double,double>> strip;
    for (int i = l; i <= r; i++)
        if (abs(pts[i].first - pts[m].first) < d) strip.push_back(pts[i]);
    sort(strip.begin(), strip.end(), [](auto& a, auto& b) { return a.second < b.second; });
    for (int i = 0; i < strip.size(); i++)
        for (int j = i+1; j < strip.size() && strip[j].second - strip[i].second < d; j++)
            d = min(d, hypot(strip[i].first-strip[j].first, strip[i].second-strip[j].second));
    return d;
}
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 说明 |
|------|-----------|------|
| 归并排序 | $O(n \log n)$ | |
| 快速幂 | $O(\log n)$ | |
| 最大子数组和 | $O(n \log n)$ | Kadane 算法 $O(n)$ |
| 最近点对 | $O(n \log n)$ | 排序+分治 |

---

## 五、竞赛与面试应用场景

1. **LeetCode 53：** 最大子数组和
2. **LeetCode 50：** Pow(x, n)
3. **LeetCode 148：** 排序链表（归并排序）
4. **LeetCode 4：** 寻找两个正序数组的中位数
5. **最近点对：** 计算几何经典问题
