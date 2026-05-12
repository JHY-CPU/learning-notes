# 7-归并排序实现 (Merge Sort Implementation)

本节展示归并排序的多种实现方式，包括递归、迭代和原地版本。

## JavaScript 实现

```javascript
// 1. 标准递归实现
function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  return merge(mergeSort(arr.slice(0, mid)), mergeSort(arr.slice(mid)));
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) result.push(left[i++]);
    else result.push(right[j++]);
  }
  return result.concat(left.slice(i), right.slice(j));
}

// 2. 原地归并（减少数组创建）
function mergeSortInPlace(arr, left = 0, right = arr.length - 1) {
  if (left >= right) return;
  const mid = Math.floor((left + right) / 2);
  mergeSortInPlace(arr, left, mid);
  mergeSortInPlace(arr, mid + 1, right);
  mergeInPlace(arr, left, mid, right);
}

function mergeInPlace(arr, left, mid, right) {
  const temp = [];
  let i = left, j = mid + 1;
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) temp.push(arr[i++]);
    else temp.push(arr[j++]);
  }
  while (i <= mid) temp.push(arr[i++]);
  while (j <= right) temp.push(arr[j++]);
  for (let k = 0; k < temp.length; k++) arr[left + k] = temp[k];
}

// 3. 自底向上迭代实现（避免递归栈）
function mergeSortIterative(arr) {
  const n = arr.length;
  for (let size = 1; size < n; size *= 2) {
    for (let l = 0; l < n; l += 2 * size) {
      const mid = Math.min(l + size - 1, n - 1);
      const r = Math.min(l + 2 * size - 1, n - 1);
      if (mid < r) mergeInPlace(arr, l, mid, r);
    }
  }
  return arr;
}

// 测试
const data = [38, 27, 43, 3, 9, 82, 10];
console.log(mergeSort([...data]));           // [3, 9, 10, 27, 38, 43, 82]
console.log(mergeSortIterative([...data]));  // [3, 9, 10, 27, 38, 43, 82]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

// 标准递归
void merge(vector<int>& a, int l, int m, int r) {
    vector<int> tmp(a.begin() + l, a.begin() + r + 1);
    int i = 0, j = m - l + 1, k = l;
    int mid = m - l;
    while (i <= mid && j <= tmp.size() - 1) {
        if (tmp[i] <= tmp[j]) a[k++] = tmp[i++];
        else a[k++] = tmp[j++];
    }
    while (i <= mid) a[k++] = tmp[i++];
    while (j < tmp.size()) a[k++] = tmp[j++];
}

void mergeSort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    mergeSort(a, l, m);
    mergeSort(a, m + 1, r);
    merge(a, l, m, r);
}

// 迭代版
void mergeSortIterative(vector<int>& a) {
    int n = a.size();
    for (int sz = 1; sz < n; sz *= 2) {
        for (int l = 0; l < n; l += 2 * sz) {
            int m = min(l + sz - 1, n - 1);
            int r = min(l + 2 * sz - 1, n - 1);
            if (m < r) merge(a, l, m, r);
        }
    }
}
```

## 三种实现对比

| 实现 | 空间 | 递归深度 | 适用场景 |
|------|------|---------|---------|
| 标准递归 | O(n log n) 总 | O(log n) | 通用 |
| 原地递归 | O(n) | O(log n) | 节省内存 |
| 迭代 | O(n) | 无 | 避免栈溢出 |

## 常见陷阱

1. **slice 开销**：JS 中 `arr.slice()` 每次创建新数组，原地版本更高效
2. **递归深度**：超大数组可能导致栈溢出，用迭代版
3. **merge 条件**：`<=` 保证稳定性
