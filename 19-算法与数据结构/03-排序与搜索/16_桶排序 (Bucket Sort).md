# 17-桶排序 (Bucket Sort)

桶排序将数据分到有限数量的桶里，每个桶内分别排序，最后合并。适用于数据均匀分布的场景。

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 平均（均匀分布） | O(n + k) | O(n + k) |
| 最坏（全部在同一桶） | O(n²) | O(n) |

稳定性：取决于桶内排序算法。

## JavaScript 实现

```javascript
function bucketSort(arr, bucketSize = 5) {
  if (arr.length <= 1) return arr;
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const bucketCount = Math.floor((max - min) / bucketSize) + 1;
  const buckets = Array.from({ length: bucketCount }, () => []);

  // 分配到桶
  for (const num of arr) {
    const idx = Math.floor((num - min) / bucketSize);
    buckets[idx].push(num);
  }

  // 每个桶内排序并合并
  const result = [];
  for (const bucket of buckets) {
    bucket.sort((a, b) => a - b); // 桶内用任意排序
    result.push(...bucket);
  }
  return result;
}

// 浮点数桶排序（0-1 范围）
function bucketSortFloat(arr) {
  const n = arr.length;
  const buckets = Array.from({ length: n }, () => []);

  for (const num of arr) {
    const idx = Math.min(Math.floor(num * n), n - 1);
    buckets[idx].push(num);
  }

  const result = [];
  for (const bucket of buckets) {
    bucket.sort((a, b) => a - b);
    result.push(...bucket);
  }
  return result;
}

console.log(bucketSort([29, 25, 3, 49, 9, 37, 21, 43]));
// [3, 9, 21, 25, 29, 37, 43, 49]
console.log(bucketSortFloat([0.42, 0.32, 0.23, 0.52, 0.25, 0.47, 0.51]));
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

void bucketSort(vector<float>& arr) {
    int n = arr.size();
    vector<vector<float>> buckets(n);
    for (float x : arr) {
        int idx = min((int)(x * n), n - 1);
        buckets[idx].push_back(x);
    }
    for (auto& b : buckets) sort(b.begin(), b.end());
    int k = 0;
    for (auto& b : buckets)
        for (float x : b) arr[k++] = x;
}
```

## 桶数量选择

| 桶数量 | 效果 |
|--------|------|
| 太少 | 桶内元素多，接近 O(n²) |
| 太多 | 空间浪费，空桶多 |
| 适中 | O(n) 均匀分布时最优 |

一般取 n 个桶或 sqrt(n) 个桶。

## 常见陷阱

1. **分布不均匀**：所有元素落入同一桶时退化为 O(n²)
2. **桶内排序选择**：小桶用插入排序，大桶用快排
3. **浮点数边界**：floor(num * n) 可能等于 n，需要 min 修正
