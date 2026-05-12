# 39-算法中的前缀和技巧 (Prefix Sum)

前缀和将区间查询转化为 O(1) 的减法操作，是预计算思想的核心体现。

## 核心思想

预处理数组的累积和，使得任意区间的和可以通过两次查表和一次减法得到。

## 适用场景

| 场景 | 方法 | 查询复杂度 |
|------|------|-----------|
| 一维区间和 | 前缀和 | O(1) |
| 二维子矩阵和 | 二维前缀和 | O(1) |
| 区间计数 | 前缀计数 | O(1) |
| 子数组和为 k | 前缀和 + 哈希 | O(n) |

## JavaScript 实现

```javascript
// 一维前缀和
class PrefixSum {
  constructor(nums) {
    this.prefix = [0];
    for (const n of nums) {
      this.prefix.push(this.prefix[this.prefix.length - 1] + n);
    }
  }

  // 查询 nums[l..r] 的和
  rangeSum(l, r) {
    return this.prefix[r + 1] - this.prefix[l];
  }
}

// 二维前缀和
class PrefixSum2D {
  constructor(matrix) {
    const m = matrix.length, n = matrix[0].length;
    this.prefix = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        this.prefix[i + 1][j + 1] = matrix[i][j]
          + this.prefix[i][j + 1]
          + this.prefix[i + 1][j]
          - this.prefix[i][j];
      }
    }
  }

  // 查询子矩阵 (r1,c1) 到 (r2,c2) 的和
  sumRegion(r1, c1, r2, c2) {
    return this.prefix[r2 + 1][c2 + 1]
      - this.prefix[r1][c2 + 1]
      - this.prefix[r2 + 1][c1]
      + this.prefix[r1][c1];
  }
}

// 和为 k 的子数组个数（LeetCode 560）
function subarraySum(nums, k) {
  const prefixCount = new Map();
  prefixCount.set(0, 1);
  let sum = 0, count = 0;
  for (const n of nums) {
    sum += n;
    if (prefixCount.has(sum - k)) count += prefixCount.get(sum - k);
    prefixCount.set(sum, (prefixCount.get(sum) || 0) + 1);
  }
  return count;
}

// 前缀异或：子数组异或为 0 的判断
function hasZeroXorSubarray(nums) {
  const seen = new Set();
  let xor = 0;
  seen.add(0);
  for (const n of nums) {
    xor ^= n;
    if (seen.has(xor)) return true;
    seen.add(xor);
  }
  return false;
}

// 测试
const ps = new PrefixSum([1, 2, 3, 4, 5]);
console.log(ps.rangeSum(1, 3));  // 9 (2+3+4)
console.log(ps.rangeSum(0, 4));  // 15

const matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]];
const ps2d = new PrefixSum2D(matrix);
console.log(ps2d.sumRegion(2, 1, 4, 3)); // 8

console.log(subarraySum([1, 1, 1], 2));   // 2
console.log(hasZeroXorSubarray([4, 2, 1, 2, 4])); // true (2^1^2=1, not 0, but 4^2^1^2^4=1, let's check)
```

## C++ 实现

```cpp
#include <vector>
#include <unordered_map>
using namespace std;

// 一维前缀和
class PrefixSum {
    vector<long long> prefix;
public:
    PrefixSum(vector<int>& nums) {
        prefix.resize(nums.size() + 1, 0);
        for (int i = 0; i < nums.size(); i++)
            prefix[i + 1] = prefix[i] + nums[i];
    }
    long long rangeSum(int l, int r) {
        return prefix[r + 1] - prefix[l];
    }
};

// 二维前缀和
class PrefixSum2D {
    vector<vector<long long>> prefix;
public:
    PrefixSum2D(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        prefix.assign(m + 1, vector<long long>(n + 1, 0));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                prefix[i + 1][j + 1] = matrix[i][j] + prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j];
    }
    long long sumRegion(int r1, int c1, int r2, int c2) {
        return prefix[r2 + 1][c2 + 1] - prefix[r1][c2 + 1] - prefix[r2 + 1][c1] + prefix[r1][c1];
    }
};

// 和为 k 的子数组个数
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixCount;
    prefixCount[0] = 1;
    int sum = 0, count = 0;
    for (int n : nums) {
        sum += n;
        if (prefixCount.count(sum - k)) count += prefixCount[sum - k];
        prefixCount[sum]++;
    }
    return count;
}
```

## 复杂度

| 操作 | 预处理 | 查询 |
|------|--------|------|
| 一维前缀和 | O(n) | O(1) |
| 二维前缀和 | O(mn) | O(1) |
| 和为 k 的子数组 | O(n) | O(n) 总计 |

## 常见陷阱

1. **越界**：prefix 数组长度为 n + 1，prefix[0] = 0
2. **溢出**：大数组前缀和可能溢出 int，用 long long
3. **二维公式**：容斥原理容易记错，记住"减去两个重叠部分加上一个交集"
4. **前缀和 + 哈希**：注意先查再更新，否则单个元素的情况会遗漏

## 实际应用

前缀和是区间查询问题的基础技巧。LeetCode 560、303、304 都是其经典应用。竞赛中频繁出现的区间和、区间计数问题几乎都需要前缀和。与哈希表结合可以解决"和为 k 的子数组"这类看似需要 O(n²) 的问题。
