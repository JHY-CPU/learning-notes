# 11-分治算法基础 (Divide and Conquer)

分治算法将问题递归地分解为更小的子问题，分别求解后合并结果。

## 三个步骤

1. **分**：将原问题拆分为若干规模更小的子问题
2. **治**：递归解决子问题（足够小时直接求解）
3. **合**：将子问题的解合并为原问题的解

```javascript
// 分治求和
function dcSum(arr, l = 0, r = arr.length - 1) {
  if (l === r) return arr[l];
  const mid = (l + r) >> 1;
  return dcSum(arr, l, mid) + dcSum(arr, mid + 1, r);
}

// 分治求最大值
function dcMax(arr, l = 0, r = arr.length - 1) {
  if (l === r) return arr[l];
  const mid = (l + r) >> 1;
  return Math.max(dcMax(arr, l, mid), dcMax(arr, mid + 1, r));
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

int dcSum(vector<int>& arr, int l, int r) {
    if (l == r) return arr[l];
    int mid = (l + r) / 2;
    return dcSum(arr, l, mid) + dcSum(arr, mid + 1, r);
}

// 归并排序
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> tmp(r - l + 1);
    int i = l, j = m + 1, k = 0;
    while (i <= m && j <= r)
        tmp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    while (i <= m) tmp[k++] = arr[i++];
    while (j <= r) tmp[k++] = arr[j++];
    for (int t = 0; t < k; t++) arr[l + t] = tmp[t];
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}
```

## 经典案例

| 算法 | 分 | 合 | 复杂度 |
|------|-----|-----|--------|
| 归并排序 | 分成两半 | 合并有序数组 | O(n log n) |
| 快速排序 | 按 pivot 分三段 | 原地拼接 | O(n log n) 均摊 |
| 最大子数组 | 左/右/跨中 | 取最大 | O(n log n) |
| 大整数乘法 | 拆分为两半 | Karatsuba | O(n^1.585) |
| 矩阵乘法 | 分成4块 | Strassen | O(n^2.807) |

## 分治 vs DP

| 特性 | 分治 | 动态规划 |
|------|------|---------|
| 子问题 | 互不重叠 | 可能重叠 |
| 递归 | 不记忆 | 记忆化或表格法 |
| 适用 | 独立分解问题 | 重叠子问题 |

## 何时使用分治

- 问题可以分解为相同类型的子问题
- 子问题之间相互独立
- 合并子问题解的代价较低
- 典型：排序、查找、矩阵运算

## 常见陷阱

1. **递归终止条件**：必须正确处理基本情况
2. **中点选择**：`(l + r) >> 1` 避免溢出
3. **合并代价**：合并步骤的复杂度决定了总复杂度
4. **栈深度**：深度过大的递归可能导致栈溢出
