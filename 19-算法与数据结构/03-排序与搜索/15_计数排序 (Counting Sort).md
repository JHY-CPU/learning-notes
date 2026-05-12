# 16-计数排序 (Counting Sort)

计数排序是一种非比较排序，通过统计每个元素出现的次数来确定位置，适用于数据范围有限的场景。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 时间 | O(n + k)，k 为数据范围 |
| 空间 | O(k) |
| 稳定性 | 稳定 |

## JavaScript 实现

```javascript
// 标准计数排序
function countingSort(arr) {
  if (arr.length <= 1) return arr;
  const max = Math.max(...arr);
  const min = Math.min(...arr);
  const range = max - min + 1;
  const count = new Array(range).fill(0);
  const output = new Array(arr.length);

  // 统计每个元素出现次数
  for (const num of arr) count[num - min]++;

  // 前缀和，确定每个元素的最终位置
  for (let i = 1; i < range; i++) count[i] += count[i - 1];

  // 从后向前遍历，保证稳定性
  for (let i = arr.length - 1; i >= 0; i--) {
    const idx = arr[i] - min;
    output[count[idx] - 1] = arr[i];
    count[idx]--;
  }
  return output;
}

// 简化版（仅计数，不保证相对顺序）
function countingSortSimple(arr) {
  const max = Math.max(...arr);
  const min = Math.min(...arr);
  const count = new Array(max - min + 1).fill(0);
  for (const num of arr) count[num - min]++;
  const result = [];
  for (let i = 0; i < count.length; i++) {
    while (count[i]-- > 0) result.push(i + min);
  }
  return result;
}

console.log(countingSort([4, 2, 2, 8, 3, 3, 1]));  // [1, 2, 2, 3, 3, 4, 8]
console.log(countingSortSimple([4, 2, 2, 8, 3, 3, 1])); // [1, 2, 2, 3, 3, 4, 8]
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

vector<int> countingSort(vector<int>& arr) {
    if (arr.empty()) return arr;
    int mn = *min_element(arr.begin(), arr.end());
    int mx = *max_element(arr.begin(), arr.end());
    vector<int> count(mx - mn + 1, 0);
    for (int x : arr) count[x - mn]++;

    vector<int> result;
    for (int i = 0; i < count.size(); i++)
        while (count[i]--) result.push_back(i + mn);
    return result;
}
```

## 算法步骤

1. 找出最大值和最小值，确定范围
2. 创建计数数组，统计每个元素出现次数
3. 前缀和累加，确定每个元素最终位置
4. 从后向前遍历原数组，放入正确位置

## 适用场景

- 数据范围小（k << n）：如年龄、成绩、字符
- 需要稳定排序
- 作为基数排序的子程序

## 常见陷阱

1. **范围太大**：k 很大时空间不可接受，如排序 [1, 1000000] 浪费大量空间
2. **负数处理**：需要减去 min 偏移
3. **稳定性**：从后向前遍历保证稳定性，从前向后不行
