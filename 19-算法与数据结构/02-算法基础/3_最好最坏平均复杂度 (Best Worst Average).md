# 04-最好最坏平均复杂度 (Best Worst Average)

同一个算法在不同输入下的性能差异可能很大。分析三种情况有助于全面理解算法表现。

## 三种情况

```javascript
// 线性查找：三种情况分析
function linearSearch(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
  }
  return -1;
}

// 最好情况 O(1)：目标在第一个位置
linearSearch([1, 2, 3, 4, 5], 1); // 立即返回

// 最坏情况 O(n)：目标在最后或不存在
linearSearch([1, 2, 3, 4, 5], 5); // 遍历全部

// 平均情况 O(n)：期望检查 n/2 个元素
// 假设等概率出现在任意位置：E = (1+2+...+n)/n = (n+1)/2
```

## C++ 示例

```cpp
// 快速排序：最好/最坏/平均
// 最好 O(n log n)：每次 pivot 平分
// 最坏 O(n²)：已排序数组，每次只分出一个元素
// 平均 O(n log n)：随机输入

#include <vector>
#include <cstdlib>
using namespace std;

int partition(vector<int>& arr, int l, int r) {
    int pivot = arr[r], i = l;
    for (int j = l; j < r; j++) {
        if (arr[j] < pivot) swap(arr[i++], arr[j]);
    }
    swap(arr[i], arr[r]);
    return i;
}

// 随机化避免最坏情况
int randomPartition(vector<int>& arr, int l, int r) {
    int idx = l + rand() % (r - l + 1);
    swap(arr[idx], arr[r]);
    return partition(arr, l, r);
}
```

## 常见算法的三种复杂度

| 算法 | 最好 | 最坏 | 平均 |
|------|------|------|------|
| 线性查找 | O(1) | O(n) | O(n) |
| 二分查找 | O(1) | O(log n) | O(log n) |
| 冒泡排序 | O(n)* | O(n²) | O(n²) |
| 快速排序 | O(n log n) | O(n²) | O(n log n) |
| 插入排序 | O(n)** | O(n²) | O(n²) |
| 哈希查找 | O(1) | O(n) | O(1) |

*优化版，无交换时提前退出
**已排序数组

## 概率分析

```javascript
// 线性查找的期望时间
// 假设目标等概率在 n 个位置中
// E = (1 + 2 + 3 + ... + n) / n = (n+1) / 2 = O(n)

// 随机化快速排序
// 随机选择 pivot，期望递归深度为 O(log n)
// 每层 O(n) 合并，总期望 O(n log n)
// 最坏情况的概率极低：P(最坏) ≈ 1/n!
```

## 随机化算法

```javascript
// 随机化快排：随机选择 pivot
function randomizedQuickSort(arr, l = 0, r = arr.length - 1) {
  if (l >= r) return;
  // 随机选择 pivot，避免最坏情况
  const randIdx = l + Math.floor(Math.random() * (r - l + 1));
  [arr[randIdx], arr[r]] = [arr[r], arr[randIdx]];

  const pivot = arr[r];
  let i = l;
  for (let j = l; j < r; j++) {
    if (arr[j] < pivot) [arr[i++], arr[j]] = [arr[j], arr[i]];
  }
  [arr[i], arr[r]] = [arr[r], arr[i]];

  randomizedQuickSort(arr, l, i - 1);
  randomizedQuickSort(arr, i + 1, r);
}
```

## 实际选择

- **实时系统**：用最坏情况分析，保证上限
- **通用算法**：用平均情况评估实际表现
- **理论分析**：最好情况帮助理解算法下界
- **面试**：三种情况都要能分析清楚
