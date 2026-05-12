# 6-归并排序基础 (Merge Sort Basics)

归并排序采用分治策略，将数组不断二分直到每个子序列只有一个元素，再两两合并成有序序列。

## 分治过程

```
[38, 27, 43, 3, 9, 82, 10]
       /              \
[38, 27, 43, 3]    [9, 82, 10]
  /       \          /      \
[38, 27]  [43, 3]  [9, 82]  [10]
 /    \    /    \   /    \
[38] [27] [43] [3] [9] [82]
  \   /    \   /     \   /
[27, 38]  [3, 43]  [9, 82]
   \       /          |
  [3, 27, 38, 43]    [9, 10, 82]
        \              /
     [3, 9, 10, 27, 38, 43, 82]
```

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 平均时间 | O(n log n) |
| 最坏时间 | O(n log n) |
| 最好时间 | O(n log n) |
| 空间 | O(n) |
| 稳定性 | 稳定 |

## JavaScript 实现

```javascript
// 基础归并排序
function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  return merge(left, right);
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

console.log(mergeSort([38, 27, 43, 3, 9, 82, 10]));
// [3, 9, 10, 27, 38, 43, 82]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (int p = 0; p < k; p++) arr[l + p] = temp[p];
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}
```

## 归并操作详解

合并两个有序数组：
- 同时遍历两个有序数组，比较当前元素
- 将较小的元素放入结果数组
- 指针后移，继续比较
- 剩余元素全部追加

## 优缺点

| 优点 | 缺点 |
|------|------|
| 稳定排序 | 需要 O(n) 额外空间 |
| 最坏也是 O(n log n) | 不是原地排序 |
| 适合链表排序 | 常数因子较大 |
| 适合外部排序 | 小数据不如插入排序 |

## 常见陷阱

1. **mid 计算**：`l + (r - l) / 2` 防止溢出
2. **合并条件**：`left[i] <= right[j]` 保证稳定性（不是 `<`）
3. **空间使用**：每次递归都创建新数组，注意内存
