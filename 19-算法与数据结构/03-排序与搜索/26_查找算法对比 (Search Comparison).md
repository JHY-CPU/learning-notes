# 27-查找算法对比 (Search Comparison)

对各种查找算法进行全面对比。

## 理论对比

| 算法 | 平均 | 最坏 | 空间 | 数据要求 |
|------|------|------|------|---------|
| 顺序查找 | O(n) | O(n) | O(1) | 无要求 |
| 二分查找 | O(log n) | O(log n) | O(1) | 有序 |
| 插值查找 | O(log log n) | O(n) | O(1) | 有序+均匀 |
| 斐波那契查找 | O(log n) | O(log n) | O(1) | 有序 |
| 指数查找 | O(log n) | O(log n) | O(1) | 有序 |
| 分块查找 | O(sqrt(n)) | O(n) | O(sqrt(n)) | 块间有序 |
| 哈希查找 | O(1) | O(n) | O(n) | 无要求 |
| 二叉搜索树 | O(log n) | O(n) | O(n) | 无要求 |
| 平衡BST | O(log n) | O(log n) | O(n) | 无要求 |

## JavaScript 实现

```javascript
// 查找算法性能对比
function benchmark() {
  const n = 1000000;
  const sortedArr = Array.from({ length: n }, (_, i) => i * 2);
  const target = sortedArr[n - 10]; // 接近末尾

  // 二分查找
  function binarySearch(arr, t) {
    let l = 0, r = arr.length - 1;
    while (l <= r) {
      const m = (l + r) >> 1;
      if (arr[m] === t) return m;
      if (arr[m] < t) l = m + 1;
      else r = m - 1;
    }
    return -1;
  }

  // 顺序查找
  function linearSearch(arr, t) {
    for (let i = 0; i < arr.length; i++) if (arr[i] === t) return i;
    return -1;
  }

  // 插值查找
  function interpolationSearch(arr, t) {
    let l = 0, r = arr.length - 1;
    while (l <= r && t >= arr[l] && t <= arr[r]) {
      if (l === r) return arr[l] === t ? l : -1;
      const pos = l + Math.floor((t - arr[l]) / (arr[r] - arr[l]) * (r - l));
      if (arr[pos] === t) return pos;
      if (arr[pos] < t) l = pos + 1;
      else r = pos - 1;
    }
    return -1;
  }

  let start = performance.now();
  binarySearch(sortedArr, target);
  console.log(`二分查找: ${(performance.now() - start).toFixed(4)}ms`);

  start = performance.now();
  interpolationSearch(sortedArr, target);
  console.log(`插值查找: ${(performance.now() - start).toFixed(4)}ms`);

  // 顺序查找太慢，跳过
  console.log(`顺序查找: 预计 ~${(n * 0.0001).toFixed(0)}ms (O(n))`);
}
```

## C++ 实现

```cpp
#include <vector>
#include <unordered_map>
#include <chrono>
using namespace std;
using namespace std::chrono;

int binarySearch(vector<int>& a, int t) {
    int l = 0, r = a.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] == t) return m;
        if (a[m] < t) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

int hashSearch(unordered_map<int, int>& mp, int t) {
    auto it = mp.find(t);
    return it != mp.end() ? it->second : -1;
}
```

## 选择建议

| 场景 | 推荐 |
|------|------|
| 无序数据 | 哈希表或顺序查找 |
| 有序 + 通用 | 二分查找 |
| 有序 + 均匀分布 | 插值查找 |
| 有序 + 目标靠前 | 指数查找 |
| 动态数据 + 需要插入 | 平衡 BST / 分块 |
| 需要 O(1) 查找 | 哈希表 |

## 常见陷阱

1. **二分查找万能论**：无序数据不能用二分
2. **哈希表无代价**：哈希表需要 O(n) 空间且有冲突
3. **插值查找不通用**：仅在均匀分布时优
4. **忽略预处理代价**：排序的 O(n log n) 可能超过直接顺序查找
