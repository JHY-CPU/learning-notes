## 05-稀疏数组 (Sparse Array)

稀疏数组是大部分元素为默认值（如0）的数组。通过只存储非默认值来节省空间。

## 稀疏数组概念

在 JavaScript 中，稀疏数组指有大量 empty 槽位的数组，或者在算法中指大部分元素值相同的数组。

```javascript

// JavaScript 中的稀疏数组（有空洞）
let sparse = [1, , , 4, , 6];
console.log(sparse.length); // 6
console.log(sparse[1]); // undefined
console.log(sparse[3]); // 4

// 避免稀疏数组的技巧
// bad: 创建时留空洞
let bad = [1, , 3];

// good: 用 undefined 占位
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
```

## 交互演示
