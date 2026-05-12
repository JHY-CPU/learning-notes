# 25-指数查找 (Exponential Search)

指数查找先通过指数增长快速缩小范围，然后在确定区间内执行二分查找。特别适合无界数据或目标靠近开头的情况。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 时间 | O(log n) |
| 空间 | O(1) |

如果目标在前 k 个元素中，时间约为 O(log k)，优于二分查找的 O(log n)。

## JavaScript 实现

```javascript
function exponentialSearch(arr, target) {
  const n = arr.length;
  if (n === 0) return -1;
  if (arr[0] === target) return 0;

  // 指数增长确定范围
  let i = 1;
  while (i < n && arr[i] <= target) {
    i *= 2;
  }

  // 在 [i/2, min(i, n-1)] 区间二分查找
  const left = Math.floor(i / 2);
  const right = Math.min(i, n - 1);
  return binarySearch(arr, target, left, right);
}

function binarySearch(arr, target, left, right) {
  while (left <= right) {
    const mid = (left + right) >> 1;
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}

// 测试
const arr = [2, 3, 4, 10, 40, 60, 70, 80, 90, 100, 110, 120];
console.log(exponentialSearch(arr, 10));   // 3
console.log(exponentialSearch(arr, 110));  // 10
console.log(exponentialSearch(arr, 55));   // -1
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int binarySearch(vector<int>& a, int target, int l, int r) {
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] == target) return m;
        if (a[m] < target) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

int exponentialSearch(vector<int>& a, int target) {
    int n = a.size();
    if (n == 0) return -1;
    if (a[0] == target) return 0;
    int i = 1;
    while (i < n && a[i] <= target) i *= 2;
    return binarySearch(a, target, i / 2, min(i, n - 1));
}
```

## 算法过程

以 arr = [2, 3, 4, 10, 40, 60, 70, 80]，target = 70 为例：
1. arr[0]=2 != 70
2. i=1: arr[1]=3 <= 70, i=2
3. i=2: arr[2]=4 <= 70, i=4
4. i=4: arr[4]=40 <= 70, i=8
5. i=8: 超出范围，区间为 [4, 7]
6. 二分查找 [40, 60, 70, 80]，找到 70

## 适用场景

- 无界数据或无限流
- 目标值靠近数组开头
- 不知道数组长度
- 二分查找前的快速范围确定

## 常见陷阱

1. **数组为空**：需要特殊处理
2. **i 溢出**：超大数组中 i 可能溢出
3. **边界**：Math.min(i, n - 1) 防止越界
