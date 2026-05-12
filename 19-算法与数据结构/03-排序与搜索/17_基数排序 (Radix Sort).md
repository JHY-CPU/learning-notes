# 18-基数排序 (Radix Sort)

基数排序是一种非比较排序，按位对数字进行排序。有 LSD（最低位优先）和 MSD（最高位优先）两种方式。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 时间 | O(d * (n + k)) |
| 空间 | O(n + k) |
| 稳定性 | 稳定 |

d 为最大数字的位数，k 为基数（十进制为 10）。

## JavaScript 实现

```javascript
// LSD 基数排序（从低位到高位）
function radixSort(arr) {
  if (arr.length <= 1) return arr;
  const max = Math.max(...arr);
  const maxDigits = String(max).length;

  for (let pos = 0; pos < maxDigits; pos++) {
    const buckets = Array.from({ length: 10 }, () => []);
    for (const num of arr) {
      const digit = Math.floor(num / Math.pow(10, pos)) % 10;
      buckets[digit].push(num);
    }
    arr = [].concat(...buckets);
  }
  return arr;
}

// 支持负数的基数排序
function radixSortWithNegatives(arr) {
  const positives = arr.filter(x => x >= 0);
  const negatives = arr.filter(x => x < 0).map(x => -x);

  const sortedPos = radixSort(positives);
  const sortedNeg = radixSort(negatives).reverse().map(x => -x);
  return [...sortedNeg, ...sortedPos];
}

// 字符串基数排序
function radixSortStrings(arr, maxLen) {
  for (let pos = maxLen - 1; pos >= 0; pos--) {
    const buckets = Array.from({ length: 256 }, () => []);
    for (const s of arr) {
      const code = pos < s.length ? s.charCodeAt(pos) : 0;
      buckets[code].push(s);
    }
    arr = [].concat(...buckets);
  }
  return arr;
}

console.log(radixSort([170, 45, 75, 90, 802, 24, 2, 66]));
// [2, 24, 45, 66, 75, 90, 170, 802]
console.log(radixSortWithNegatives([-5, 3, -2, 8, 1, -7]));
// [-7, -5, -2, 1, 3, 8]
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

void radixSort(vector<int>& arr) {
    int maxVal = *max_element(arr.begin(), arr.end());
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        vector<vector<int>> buckets(10);
        for (int x : arr) buckets[(x / exp) % 10].push_back(arr);
        int k = 0;
        for (auto& b : buckets)
            for (int x : b) arr[k++] = x;
    }
}
```

## LSD vs MSD

| 特性 | LSD | MSD |
|------|-----|-----|
| 处理顺序 | 低位到高位 | 高位到低位 |
| 实现 | 简单（迭代） | 复杂（递归） |
| 适用 | 定长数据 | 变长数据（字符串） |
| 稳定性 | 需要稳定子排序 | 自然适合递归 |

## 适用场景

- 整数排序且范围已知
- 字符串排序（MSD 基数排序）
- IP 地址、日期等固定格式数据

## 常见陷阱

1. **负数**：标准基数排序不支持负数，需分离处理
2. **浮点数**：不能直接用基数排序
3. **位数 d 大时**：d 很大时不如比较排序
