# 差分数组专题 (Difference Array)

## 一、概念定义与原理

### 1.1 差分数组定义

对于数组 $a[0 \ldots n-1]$，差分数组 $d[i]$ 定义为：

$$d[i] = \begin{cases} a[0] & i = 0 \\ a[i] - a[i-1] & i > 0 \end{cases}$$

数组 $a$ 是差分数组 $d$ 的前缀和。

### 1.2 核心性质

**区间加：** 将 $a[l \ldots r]$ 每个元素加 $v$：
- $d[l] += v$
- $d[r+1] -= v$（如果 $r+1 < n$）

**还原原数组：** 对 $d$ 做前缀和即可得到原数组。

### 1.3 二维差分

对子矩阵 $[r_1, c_1]$ 到 $[r_2, c_2]$ 每个元素加 $v$：
- $d[r_1][c_1] += v$
- $d[r_1][c_2+1] -= v$
- $d[r_2+1][c_1] -= v$
- $d[r_2+1][c_2+1] += v$

---

## 二、核心算法

### 2.1 一维差分

1. 构造差分数组
2. 对每次区间加操作 $O(1)$ 更新
3. 最后做前缀和还原

### 2.2 二维差分

1. 构造差分矩阵
2. 对每次子矩阵加操作 $O(1)$ 更新
3. 最后做二维前缀和还原

---

## 三、代码实现

### 3.1 一维差分 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

class DifferenceArray {
    vector<long long> diff;
    int n;
public:
    DifferenceArray(vector<int>& a) {
        n = a.size();
        diff.resize(n + 1, 0);
        diff[0] = a[0];
        for (int i = 1; i < n; i++) diff[i] = a[i] - a[i-1];
    }

    // 将 [l, r] 区间每个元素加 val
    void range_add(int l, int r, long long val) {
        diff[l] += val;
        if (r + 1 <= n) diff[r + 1] -= val;
    }

    // 获取还原后的数组
    vector<long long> build() {
        vector<long long> result(n);
        result[0] = diff[0];
        for (int i = 1; i < n; i++) result[i] = result[i-1] + diff[i];
        return result;
    }
};
```

### 3.2 二维差分 - C++

```cpp
class DiffArray2D {
    vector<vector<long long>> diff;
    int m, n;
public:
    DiffArray2D(vector<vector<int>>& a) {
        m = a.size(); n = a[0].size();
        diff.assign(m + 1, vector<long long>(n + 1, 0));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                diff[i][j] += a[i][j];
                diff[i][j+1] -= a[i][j];
                diff[i+1][j] -= a[i][j];
                diff[i+1][j+1] += a[i][j];
            }
    }

    void range_add(int r1, int c1, int r2, int c2, long long val) {
        diff[r1][c1] += val;
        diff[r1][c2+1] -= val;
        diff[r2+1][c1] -= val;
        diff[r2+1][c2+1] += val;
    }

    vector<vector<long long>> build() {
        vector<vector<long long>> result(m, vector<long long>(n));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                result[i][j] = diff[i][j];
                if (i > 0) result[i][j] += result[i-1][j];
                if (j > 0) result[i][j] += result[i][j-1];
                if (i > 0 && j > 0) result[i][j] -= result[i-1][j-1];
            }
        return result;
    }
};
```

### 3.3 Python 实现

```python
class DifferenceArray:
    def __init__(self, a):
        self.n = len(a)
        self.diff = [0] * (self.n + 1)
        self.diff[0] = a[0]
        for i in range(1, self.n):
            self.diff[i] = a[i] - a[i-1]

    def range_add(self, l, r, val):
        self.diff[l] += val
        if r + 1 < self.n:
            self.diff[r + 1] -= val

    def build(self):
        result = [0] * self.n
        result[0] = self.diff[0]
        for i in range(1, self.n):
            result[i] = result[i-1] + self.diff[i]
        return result

# 测试
da = DifferenceArray([1, 2, 3, 4, 5])
da.range_add(1, 3, 10)  # [1, 12, 13, 14, 5]
print(da.build())        # [1, 12, 13, 14, 5]
```

### 3.4 航班预订统计（LeetCode 1109）

```cpp
// bookings[i] = [first_i, last_i, seats_i]
// 对 [first_i-1, last_i-1] 区间加 seats_i
vector<long long> corp_flight_bookings(vector<vector<int>>& bookings, int n) {
    vector<long long> diff(n + 1, 0);
    for (auto& b : bookings) {
        diff[b[0]-1] += b[2];
        if (b[1] < n) diff[b[1]] -= b[2];
    }
    vector<long long> result(n);
    result[0] = diff[0];
    for (int i = 1; i < n; i++) result[i] = result[i-1] + diff[i];
    return result;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 构建差分数组 | $O(n)$ | $O(n)$ |
| 区间加 | $O(1)$ | - |
| 还原原数组 | $O(n)$ | $O(n)$ |
| 二维区间加 | $O(1)$ | $O(mn)$ |
| 二维还原 | $O(mn)$ | $O(mn)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 1109：** 航班预订统计
2. **LeetCode 370：** 区间加法
3. **竞赛场次统计：** 多个区间加操作后查询
4. **图像处理：** 区域亮度调整
5. **配合扫描线：** 解决矩形覆盖面积等问题
