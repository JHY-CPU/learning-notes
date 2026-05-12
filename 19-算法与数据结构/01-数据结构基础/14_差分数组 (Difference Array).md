# 15-差分数组 (Difference Array)

差分数组是前缀和的逆运算，专门用于高效处理数组的区间批量增减操作。

## 差分数组原理

```javascript
// 原始数组 arr
// 差分数组 diff: diff[i] = arr[i] - arr[i-1] (i > 0), diff[0] = arr[0]

// 核心性质：对原数组区间 [l, r] 加 val
// 只需 diff[l] += val 和 diff[r + 1] -= val
// 最后对差分数组求前缀和还原原数组

class Difference {
  constructor(nums) {
    this.diff = new Array(nums.length);
    this.diff[0] = nums[0];
    for (let i = 1; i < nums.length; i++) {
      this.diff[i] = nums[i] - nums[i - 1];
    }
  }

  increment(l, r, val) {
    this.diff[l] += val;
    if (r + 1 < this.diff.length) this.diff[r + 1] -= val;
  }

  getResult() {
    let res = new Array(this.diff.length);
    res[0] = this.diff[0];
    for (let i = 1; i < this.diff.length; i++) {
      res[i] = res[i - 1] + this.diff[i];
    }
    return res;
  }
}
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

class DifferenceArray {
    vector<int> diff;
public:
    DifferenceArray(vector<int>& nums) {
        diff.resize(nums.size());
        diff[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            diff[i] = nums[i] - nums[i - 1];
        }
    }

    void increment(int l, int r, int val) {
        diff[l] += val;
        if (r + 1 < diff.size()) diff[r + 1] -= val;
    }

    vector<int> getResult() {
        vector<int> res(diff.size());
        res[0] = diff[0];
        for (int i = 1; i < diff.size(); i++) {
            res[i] = res[i - 1] + diff[i];
        }
        return res;
    }
};
```

## 应用：航班预订统计

```javascript
// bookings = [[1,2,10], [2,3,20], [2,5,25]]
// n = 5 个航班
function corpFlightBookings(bookings, n) {
  let diff = new Array(n + 1).fill(0);
  for (let [l, r, val] of bookings) {
    diff[l] += val;
    if (r + 1 <= n) diff[r + 1] -= val;
  }
  let result = new Array(n);
  result[0] = diff[1];
  for (let i = 1; i < n; i++) {
    result[i] = result[i - 1] + diff[i + 1];
  }
  return result;
}
```

## 应用：区间加操作

```javascript
// 给定数组，执行多次区间加操作，返回最终数组
function getModifiedArray(length, updates) {
  let diff = new Array(length).fill(0);
  for (let [l, r, val] of updates) {
    diff[l] += val;
    if (r + 1 < length) diff[r + 1] -= val;
  }
  let result = [diff[0]];
  for (let i = 1; i < length; i++) {
    result[i] = result[i - 1] + diff[i];
  }
  return result;
}
```

## 二维差分

```javascript
// 二维差分：对子矩阵进行批量加操作
class DiffMatrix {
  constructor(matrix) {
    this.m = matrix.length;
    this.n = matrix[0].length;
    this.diff = Array.from({length: this.m + 1}, () => new Array(this.n + 1).fill(0));
  }

  // 对子矩阵 (r1,c1) 到 (r2,c2) 加 val
  increment(r1, c1, r2, c2, val) {
    this.diff[r1][c1] += val;
    this.diff[r1][c2 + 1] -= val;
    this.diff[r2 + 1][c1] -= val;
    this.diff[r2 + 1][c2 + 1] += val;
  }

  // 通过二维前缀和还原
  getResult() {
    let result = Array.from({length: this.m}, () => new Array(this.n).fill(0));
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        result[i][j] = this.diff[i][j]
          + (i > 0 ? result[i-1][j] : 0)
          + (j > 0 ? result[i][j-1] : 0)
          - (i > 0 && j > 0 ? result[i-1][j-1] : 0);
      }
    }
    return result;
  }
}
```

## 复杂度分析

| 操作 | 差分数组 | 暴力 |
|------|---------|------|
| 区间加操作 | O(1) | O(n) |
| m 次操作 | O(m) | O(mn) |
| 还原结果 | O(n) | - |
| 空间 | O(n) | O(n) |

## 何时使用差分数组

- 多次区间增减操作后需要最终结果
- 航班预订、会议室预订等场景
- 区间覆盖计数
- 二维矩阵区域操作

## 常见陷阱

1. **数组索引**：注意 `r + 1` 是否越界
2. **初始化**：差分数组需要从原数组计算，不能直接用全零
3. **还原顺序**：必须按顺序做前缀和还原
4. **二维差分**：四个角的加减符号容易搞混
