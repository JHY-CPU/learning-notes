# 01-算法复杂度基础 (Complexity Basics)

衡量算法效率的指标：时间复杂度（操作次数）和空间复杂度（内存使用）。

## 常见复杂度

```javascript
// O(1) - 常数时间
function constantTime(arr) { return arr[0]; }

// O(log n) - 对数时间（二分查找）
function logarithmicTime(n) {
  let c = 0;
  for (let i = 1; i < n; i *= 2) c++;
  return c;
}

// O(n) - 线性时间
function linearTime(arr) {
  let s = 0;
  for (const x of arr) s += x;
  return s;
}

// O(n log n) - 线性对数
function nlogn(n) {
  let c = 0;
  for (let i = 0; i < n; i++)
    for (let j = 1; j < n; j *= 2) c++;
  return c;
}

// O(n^2) - 平方时间
function quadraticTime(n) {
  let c = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) c++;
  return c;
}
```

## C++ 示例

```cpp
#include <iostream>
#include <vector>
using namespace std;

// O(n) 线性
int linearSearch(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++)
        if (arr[i] == target) return i;
    return -1;
}

// O(log n) 二分
int binarySearch(vector<int>& arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    return -1;
}
```

## 复杂度增长对比

| 复杂度 | 名称 | n=10 | n=100 | n=1000000 |
|--------|------|------|-------|-----------|
| O(1) | 常数 | 1 | 1 | 1 |
| O(log n) | 对数 | 3 | 7 | 20 |
| O(n) | 线性 | 10 | 100 | 10^6 |
| O(n log n) | 线性对数 | 30 | 700 | 2×10^7 |
| O(n²) | 平方 | 100 | 10000 | 10^12 |
| O(2^n) | 指数 | 1024 | 10^30 | 不可行 |

## 空间复杂度

```javascript
// O(1) 空间：原地操作
function reverseInPlace(arr) {
  let l = 0, r = arr.length - 1;
  while (l < r) [arr[l++], arr[r--]] = [arr[r], arr[l]];
}

// O(n) 空间：需要额外数组
function reverseCopy(arr) {
  return [...arr].reverse();
}

// O(n²) 空间：二维数组
function createMatrix(n) {
  return Array.from({length: n}, () => new Array(n).fill(0));
}
```

## 判断技巧

1. **单层循环**：O(n)
2. **循环中变量倍增/倍减**：O(log n)
3. **嵌套循环**：各层相乘
4. **递归**：看递推式 T(n) = aT(n/b) + f(n)
5. **排序**：O(n log n) 通常是最快排序

## 常见陷阱

1. 忽略隐含的循环（如 indexOf 内部是 O(n)）
2. 字符串拼接在某些语言中是 O(n) 而非 O(1)
3. HashMap 最坏情况 O(n)
4. 递归的空间复杂度要算上栈空间
