# 04-多维数组 (Multidimensional Arrays)

多维数组是数组的数组，用于表示矩阵、网格、图像等结构化数据。

## 创建多维数组

```javascript
// 创建 3x3 矩阵
let matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

// 访问元素 matrix[行][列]
console.log(matrix[1][2]); // 6（第2行第3列）

// 创建指定大小的二维数组
function createMatrix(rows, cols, fill = 0) {
  return Array.from({length: rows}, () => new Array(cols).fill(fill));
}

// 三维数组
let cube = [
  [[1,2],[3,4]],
  [[5,6],[7,8]]
];
console.log(cube[1][0][1]); // 6
```

## C++ 多维数组

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    // 静态二维数组
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // 动态二维数组
    int rows = 3, cols = 4;
    vector<vector<int>> grid(rows, vector<int>(cols, 0));

    // 访问
    grid[1][2] = 5;
    cout << grid[1][2] << endl;

    // 一维数组模拟二维（缓存更友好）
    vector<int> flat(rows * cols, 0);
    flat[i * cols + j] = value; // 等价于 grid[i][j]

    return 0;
}
```

## 常见遍历方式

```javascript
let matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

// 行优先遍历
for (let i = 0; i < matrix.length; i++) {
  for (let j = 0; j < matrix[i].length; j++) {
    console.log(matrix[i][j]);
  }
}

// 列优先遍历
for (let j = 0; j < matrix[0].length; j++) {
  for (let i = 0; i < matrix.length; i++) {
    console.log(matrix[i][j]);
  }
}

// 对角线遍历
for (let i = 0; i < matrix.length; i++) {
  console.log(matrix[i][i]); // 主对角线
  console.log(matrix[i][matrix.length-1-i]); // 副对角线
}

// Z 字形遍历（蛇形遍历）
function zigzagTraverse(matrix) {
  let result = [];
  let rows = matrix.length, cols = matrix[0].length;
  for (let d = 0; d < rows + cols - 1; d++) {
    let r = d < cols ? 0 : d - cols + 1;
    let c = d < cols ? d : cols - 1;
    let line = [];
    while (r < rows && c >= 0) {
      line.push(matrix[r][c]);
      r++; c--;
    }
    if (d % 2 === 0) line.reverse();
    result.push(...line);
  }
  return result;
}
```

## 矩阵转置

```javascript
// 转置：行列互换
function transpose(matrix) {
  let rows = matrix.length, cols = matrix[0].length;
  let result = createMatrix(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

// 原地转置（方阵）
function transposeInPlace(matrix) {
  let n = matrix.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
    }
  }
}
```

## 矩阵旋转

```javascript
// 顺时针旋转90度：转置 + 每行反转
function rotate90(matrix) {
  let n = matrix.length;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
  for (let i = 0; i < n; i++)
    matrix[i].reverse();
}
```

## 一维数组模拟二维

在某些场景下，用一维数组模拟二维更高效：

```javascript
// 一维数组索引映射
// matrix[i][j] 对应 flat[i * cols + j]
class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Array(rows * cols).fill(0);
  }
  get(i, j) { return this.data[i * this.cols + j]; }
  set(i, j, val) { this.data[i * this.cols + j] = val; }
}
```

优势：内存连续，缓存友好，避免多层指针跳转。

## 应用场景

- 图像处理：像素矩阵
- 游戏开发：地图网格、棋盘
- 动态规划：DP 表
- 图论：邻接矩阵
- 线性代数：矩阵运算
