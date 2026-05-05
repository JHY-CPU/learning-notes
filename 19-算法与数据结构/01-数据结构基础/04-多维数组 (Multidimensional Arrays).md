## 04-多维数组 (Multidimensional Arrays)

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

// 创建指定大小的二维数组（用循环）
function createMatrix(rows, cols, fill = 0) {
  return Array.from({length: rows}, () =>
    new Array(cols).fill(fill)
  );
}

// 三维数组
let cube = [
  [[1,2],[3,4]],
  [[5,6],[7,8]]
];
console.log(cube[1][0][1]); // 6
```

## 常见操作：矩阵遍历

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
```

## 交互演示
