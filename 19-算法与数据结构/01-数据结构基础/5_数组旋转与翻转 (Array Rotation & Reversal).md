# 06-数组旋转与翻转 (Array Rotation & Reversal)

数组旋转是指将数组元素循环左移或右移。数组翻转是指将数组元素顺序反转。

## 三次反转法旋转

旋转的核心思路是使用三次反转法实现 O(1) 额外空间：

```javascript
// 将数组右移 k 位
// 例如: [1,2,3,4,5] 右移2位 -> [4,5,1,2,3]
function rotate(nums, k) {
  k = k % nums.length;
  reverse(nums, 0, nums.length - 1);
  reverse(nums, 0, k - 1);
  reverse(nums, k, nums.length - 1);
}

function reverse(nums, start, end) {
  while (start < end) {
    [nums[start], nums[end]] = [nums[end], nums[start]];
    start++;
    end--;
  }
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// 三次反转法
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k %= n;
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.end());
}

// 环状替换法 O(n) 时间 O(1) 空间
void rotateCyclic(vector<int>& nums, int k) {
    int n = nums.size();
    k %= n;
    int count = 0;
    for (int start = 0; count < n; start++) {
        int current = start;
        int prev = nums[start];
        do {
            int next = (current + k) % n;
            int temp = nums[next];
            nums[next] = prev;
            prev = temp;
            current = next;
            count++;
        } while (current != start);
    }
}
```

## 数组翻转

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

```javascript
// 顺时针旋转 90 度
function rotateMatrix(matrix) {
  let n = matrix.length;
  // 1. 转置
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
  // 2. 每行反转
  for (let i = 0; i < n; i++) matrix[i].reverse();
}

// 逆时针旋转 90 度 = 转置 + 每列反转
function rotateMatrixCCW(matrix) {
  let n = matrix.length;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
  for (let j = 0; j < n; j++) {
    let top = 0, bottom = n - 1;
    while (top < bottom) {
      [matrix[top][j], matrix[bottom][j]] = [matrix[bottom][j], matrix[top][j]];
      top++; bottom--;
    }
  }
}
```

## 旋转数组搜索

在旋转排序数组中搜索目标值：

```javascript
// [4,5,6,7,0,1,2] 中搜索 0 -> 返回 4
function search(nums, target) {
  let left = 0, right = nums.length - 1;
  while (left <= right) {
    let mid = (left + right) >> 1;
    if (nums[mid] === target) return mid;
    if (nums[left] <= nums[mid]) {
      if (target >= nums[left] && target < nums[mid]) right = mid - 1;
      else left = mid + 1;
    } else {
      if (target > nums[mid] && target <= nums[right]) left = mid + 1;
      else right = mid - 1;
    }
  }
  return -1;
}
```

## 时间复杂度总结

| 操作 | 时间 | 空间 |
|------|------|------|
| 三次反转旋转 | O(n) | O(1) |
| 环状替换旋转 | O(n) | O(1) |
| 额外数组旋转 | O(n) | O(n) |
| 数组反转 | O(n) | O(1) |
| 矩阵旋转90度 | O(n²) | O(1) 原地 |

## 常见应用

- 轮播图数据循环
- 循环队列实现
- 滑动窗口问题预处理
- 图像旋转处理
