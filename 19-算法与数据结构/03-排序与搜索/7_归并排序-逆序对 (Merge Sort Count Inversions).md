# 8-归并排序求逆序对 (Count Inversions)

逆序对是指数组中满足 `i < j` 且 `arr[i] > arr[j]` 的元素对。利用归并排序的合并过程，可以在 O(n log n) 时间内高效统计逆序对数量。

## 核心思想

在合并两个有序子数组时，如果左子数组的元素 `arr[i] > arr[j]`（右子数组当前元素），则左子数组中从 i 到 mid 的所有元素都与 arr[j] 构成逆序对，共 `mid - i + 1` 个。

## JavaScript 实现

```javascript
function countInversions(arr) {
  if (arr.length <= 1) return { count: 0, sorted: arr };
  const mid = Math.floor(arr.length / 2);
  const left = countInversions(arr.slice(0, mid));
  const right = countInversions(arr.slice(mid));
  const merged = mergeCount(left.sorted, right.sorted);
  return {
    count: left.count + right.count + merged.count,
    sorted: merged.sorted
  };
}

function mergeCount(left, right) {
  const sorted = [];
  let i = 0, j = 0, count = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      sorted.push(left[i++]);
    } else {
      count += left.length - i; // 关键：left[i..end] 都与 right[j] 构成逆序对
      sorted.push(right[j++]);
    }
  }
  while (i < left.length) sorted.push(left[i++]);
  while (j < right.length) sorted.push(right[j++]);
  return { count, sorted };
}

// 原地版本
function countInversionsInPlace(arr) {
  let count = 0;
  function mergeSort(l, r) {
    if (l >= r) return;
    const m = (l + r) >> 1;
    mergeSort(l, m);
    mergeSort(m + 1, r);
    const temp = [];
    let i = l, j = m + 1;
    while (i <= m && j <= r) {
      if (arr[i] <= arr[j]) {
        temp.push(arr[i++]);
      } else {
        count += m - i + 1; // 逆序对数量
        temp.push(arr[j++]);
      }
    }
    while (i <= m) temp.push(arr[i++]);
    while (j <= r) temp.push(arr[j++]);
    for (let k = 0; k < temp.length; k++) arr[l + k] = temp[k];
  }
  mergeSort(0, arr.length - 1);
  return count;
}

console.log(countInversions([2, 4, 1, 3, 5]).count);  // 3
console.log(countInversionsInPlace([2, 4, 1, 3, 5]));  // 3
// 逆序对: (2,1), (4,1), (4,3)
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

long long mergeCount(vector<int>& arr, int l, int m, int r) {
    vector<int> temp;
    int i = l, j = m + 1;
    long long cnt = 0;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) {
            temp.push_back(arr[i++]);
        } else {
            cnt += m - i + 1;
            temp.push_back(arr[j++]);
        }
    }
    while (i <= m) temp.push_back(arr[i++]);
    while (j <= r) temp.push_back(arr[j++]);
    for (int k = 0; k < temp.size(); k++) arr[l + k] = temp[k];
    return cnt;
}

long long countInversions(vector<int>& arr, int l, int r) {
    if (l >= r) return 0;
    int m = l + (r - l) / 2;
    long long cnt = countInversions(arr, l, m) + countInversions(arr, m + 1, r);
    cnt += mergeCount(arr, l, m, r);
    return cnt;
}
```

## 复杂度

| 操作 | 时间 | 空间 |
|------|------|------|
| 统计逆序对 | O(n log n) | O(n) |
| 暴力方法 | O(n²) | O(1) |

## 应用场景

- 量化数组的"无序程度"
- 计算排序所需的最小交换次数（某些情况下）
- 竞赛中的逆序对计数问题

## 常见陷阱

1. **计数公式**：`m - i + 1` 不是 `m - i`，包含 arr[i] 自身
2. **溢出**：大数组逆序对数量可能很大，用 long long
3. **全局变量 vs 返回值**：注意 count 的传递方式
