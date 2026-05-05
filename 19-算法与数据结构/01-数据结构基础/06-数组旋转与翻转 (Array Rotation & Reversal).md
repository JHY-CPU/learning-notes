## 06-数组旋转与翻转 (Array Rotation & Reversal)

数组旋转是指将数组元素循环左移或右移。数组翻转是指将数组元素顺序反转。

## 数组旋转

旋转的核心思路是使用三次反转法实现 O(1) 额外空间：

```javascript

// 将数组右移 k 位
// 例如: [1,2,3,4,5] 右移2位 -> [4,5,1,2,3]

function rotate(nums, k) {
  k = k % nums.length; // 处理 k > 长度的情况
  // 三次反转法
  reverse(nums, 0, nums.length - 1); // 整体反转
  reverse(nums, 0, k - 1);           // 反转前 k 个
  reverse(nums, k, nums.length - 1); // 反转剩余部分
}

function reverse(nums, start, end) {
  while (start < end) {
    [nums[start], nums[end]] = [nums[end], nums[start]];
    start++;
    end--;
  }
}
```

## 数组翻转 / 反转

```javascript

// 反转整个数组
function reverseArray(arr) {
  let left = 0, right = arr.length - 1;
  while (left < right) {
    [arr[left], arr[right]] = [arr[right], arr[left]];
    left++;
    right--;
  }
  return arr;
}

// 反转指定区段
function reverseRange(arr, start, end) {
  while (start < end) {
    [arr[start], arr[end]] = [arr[end], arr[start]];
    start++;
    end--;
  }
}
```

## 矩阵旋转

将二维矩阵顺时针旋转90度：

```javascript

// 矩阵顺时针旋转 90 度
function rotateMatrix(matrix) {
  let n = matrix.length;
  // 1. 转置（行列互换）
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
    }
  }
  // 2. 每行反转
  for (let i = 0; i < n; i++) {
    matrix[i].reverse();
  }
}
```

## 交互演示
