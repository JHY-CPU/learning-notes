# 05-稀疏数组 (Sparse Array)

稀疏数组是大部分元素为默认值（如0）的数组。通过只存储非默认值来节省空间。

## 稀疏数组概念

在 JavaScript 中，稀疏数组指有大量 empty 槽位的数组，或者在算法中指大部分元素值相同的数组。

```javascript
// JavaScript 中的稀疏数组（有空洞）
let sparse = [1, , , 4, , 6];
console.log(sparse.length); // 6
console.log(sparse[1]); // undefined

// 避免稀疏数组的技巧
let bad = [1, , 3];
let good = [1, undefined, 3];
```

## 稀疏矩阵压缩存储

对于二维稀疏数组（稀疏矩阵），用坐标-值对来存储：

```javascript
// 压缩存储表示法：[[行,列,值], ...]
// 原始矩阵（5x5，大部分为0）：
// 0 0 0 0 0
// 0 1 0 0 0
// 0 0 0 2 0
// 0 0 0 0 0
// 0 0 3 0 0

// 压缩后只存非零元素：
let compressed = [
  [1, 1, 1],  // 第2行第2列=1
  [2, 3, 2],  // 第3行第4列=2
  [4, 2, 3]   // 第5行第3列=3
];

// 压缩算法：遍历矩阵，收集非零元素
function compress(matrix) {
  let result = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (matrix[i][j] !== 0) {
        result.push([i, j, matrix[i][j]]);
      }
    }
  }
  return result;
}

// 还原算法
function decompress(compressed, rows, cols) {
  let matrix = Array.from({length: rows}, () => new Array(cols).fill(0));
  for (let [r, c, val] of compressed) {
    matrix[r][c] = val;
  }
  return matrix;
}
```

## C++ 实现

```cpp
#include <vector>
#include <map>
using namespace std;

// 三元组表示法
struct SparseMatrix {
    int rows, cols;
    vector<tuple<int,int,int>> data; // (row, col, value)

    void add(int r, int c, int val) {
        if (val != 0) data.emplace_back(r, c, val);
    }

    int get(int r, int c) const {
        for (auto& [row, col, val] : data) {
            if (row == r && col == c) return val;
        }
        return 0;
    }
};

// 使用 map 的稀疏数组
class SparseArray {
    map<int, int> data; // index -> value
    int defaultVal;
public:
    SparseArray(int def = 0) : defaultVal(def) {}
    void set(int idx, int val) {
        if (val == defaultVal) data.erase(idx);
        else data[idx] = val;
    }
    int get(int idx) const {
        auto it = data.find(idx);
        return it != data.end() ? it->second : defaultVal;
    }
};
```

## 压缩率分析

假设 m×n 矩阵中有 k 个非零元素：

- 原始存储：m × n 个元素
- 压缩存储：k × 3 个元素（行、列、值）
- 当 k << m×n 时压缩效果显著
- 临界点：k > m×n/3 时不值得压缩

## 稀疏矩阵运算

```javascript
// 稀疏矩阵加法
function sparseAdd(A, B) {
  let map = new Map();
  for (let [r, c, v] of A) map.set(`${r},${c}`, v);
  for (let [r, c, v] of B) {
    let key = `${r},${c}`;
    map.set(key, (map.get(key) || 0) + v);
  }
  return [...map.entries()].map(([k, v]) => {
    let [r, c] = k.split(',').map(Number);
    return [r, c, v];
  }).filter(([,, v]) => v !== 0);
}

// 稀疏矩阵乘法
function sparseMultiply(A, B, n) {
  let result = new Map();
  for (let [r1, c1, v1] of A) {
    for (let [r2, c2, v2] of B) {
      if (c1 === r2) {
        let key = `${r1},${c2}`;
        result.set(key, (result.get(key) || 0) + v1 * v2);
      }
    }
  }
  return [...result.entries()].map(([k, v]) => {
    let [r, c] = k.split(',').map(Number);
    return [r, c, v];
  });
}
```

## 应用场景

- **图像处理**：稀疏像素图
- **科学计算**：大型稀疏线性方程组
- **搜索引擎**：倒排索引（词->文档列表）
- **社交网络**：邻接矩阵稀疏化
- **游戏存档**：棋盘状态只存有棋子的位置
