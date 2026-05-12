# 13-前缀和 (Prefix Sum)

前缀和是一种预处理技术，通过预先计算数组每个位置之前所有元素的和，将区间求和查询优化到 O(1)。

## 一维前缀和

```javascript
// 构建前缀和数组
// prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
function buildPrefixSum(nums) {
  let prefix = new Array(nums.length + 1).fill(0);
  for (let i = 0; i < nums.length; i++) {
    prefix[i + 1] = prefix[i] + nums[i];
  }
  return prefix;
}

// 查询区间 [l, r] 的和（包含两端）
function rangeSum(prefix, l, r) {
  return prefix[r + 1] - prefix[l];
}

// 示例：
// nums = [1, 2, 3, 4, 5]
// prefix = [0, 1, 3, 6, 10, 15]
// 区间 [1, 3] 的和 = prefix[4] - prefix[1] = 10 - 1 = 9
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

class PrefixSum {
    vector<int> prefix;
public:
    PrefixSum(vector<int>& nums) {
        prefix.resize(nums.size() + 1, 0);
        for (int i = 0; i < nums.size(); i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }
    }
    // 查询 [l, r] 区间和
    int query(int l, int r) {
        return prefix[r + 1] - prefix[l];
    }
};
```

## 二维前缀和

```javascript
// 二维前缀和：prefix[i+1][j+1] 表示 (0,0) 到 (i,j) 的矩阵和
function buildPrefixSum2D(matrix) {
  let m = matrix.length, n = matrix[0].length;
  let prefix = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      prefix[i + 1][j + 1] = prefix[i][j + 1] + prefix[i + 1][j]
                            - prefix[i][j] + matrix[i][j];
    }
  }
  return prefix;
}

// 查询子矩阵 (r1,c1) 到 (r2,c2) 的和
function submatrixSum(prefix, r1, c1, r2, c2) {
  return prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1]
       - prefix[r2 + 1][c1] + prefix[r1][c1];
}
```

## 前缀和 + 哈希表

```javascript
// 统计和为 k 的子数组个数
function subarraySum(nums, k) {
  let map = new Map();
  map.set(0, 1);
  let count = 0, prefixSum = 0;

  for (let num of nums) {
    prefixSum += num;
    if (map.has(prefixSum - k)) count += map.get(prefixSum - k);
    map.set(prefixSum, (map.get(prefixSum) || 0) + 1);
  }
  return count;
}
```

## 时间复杂度分析

| 操作 | 前缀和 | 暴力 |
|------|--------|------|
| 预处理 | O(n) | - |
| 单次查询 | O(1) | O(n) |
| m 次查询 | O(n + m) | O(nm) |

## 常见应用

- 快速区间求和查询
- 子数组和等于 k 的计数
- 二维矩阵区域和查询
- 数据分析中的滑动统计

## 常见陷阱

1. **前缀和数组长度**：通常比原数组多 1（prefix[0] = 0）
2. **边界索引**：查询 [l, r] 用 `prefix[r+1] - prefix[l]`
3. **溢出问题**：大数求和可能溢出，注意数据类型
4. **二维容斥**：减去两个子矩阵后要加回交集
