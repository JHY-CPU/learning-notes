## 13-前缀和 (Prefix Sum)

前缀和是一种预处理技术，通过预先计算数组每个位置之前所有元素的和，将区间求和查询优化到 O(1)。

## 一维前缀和

```javascript

// 构建前缀和数组
// prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
// 即 prefix[0] = 0, prefix[1] = nums[0], ...

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

## 交互演示
