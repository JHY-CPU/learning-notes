# 23-插值查找 (Interpolation Search)

插值查找是二分查找的改进版，基于数据均匀分布的假设，通过公式预测目标位置。

## 核心公式

```
pos = low + floor((target - arr[low]) / (arr[high] - arr[low]) * (high - low))
```

类似在字典中查词：不会从中间翻，而是根据首字母位置大致定位。

## 复杂度分析

| 情况 | 时间 |
|------|------|
| 均匀分布 | O(log log n) |
| 最坏（极度不均匀） | O(n) |

空间 O(1)。要求数据有序且分布均匀。

## JavaScript 实现

```javascript
function interpolationSearch(arr, target) {
  let low = 0, high = arr.length - 1;

  while (low <= high && target >= arr[low] && target <= arr[high]) {
    if (low === high) {
      return arr[low] === target ? low : -1;
    }

    // 插值公式估算位置
    const pos = low + Math.floor(
      ((target - arr[low]) / (arr[high] - arr[low])) * (high - low)
    );

    if (arr[pos] === target) return pos;
    if (arr[pos] < target) low = pos + 1;
    else high = pos - 1;
  }
  return -1;
}

// 测试
const uniform = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
console.log(interpolationSearch(uniform, 70)); // 6

const nonUniform = [1, 2, 3, 4, 5, 100, 200, 1000, 10000, 100000];
console.log(interpolationSearch(nonUniform, 100)); // 5
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int interpolationSearch(vector<int>& arr, int target) {
    int low = 0, high = arr.size() - 1;
    while (low <= high && target >= arr[low] && target <= arr[high]) {
        if (low == high) return arr[low] == target ? low : -1;
        int pos = low + (double)(target - arr[low]) / (arr[high] - arr[low]) * (high - low);
        if (arr[pos] == target) return pos;
        if (arr[pos] < target) low = pos + 1;
        else high = pos - 1;
    }
    return -1;
}
```

## 与二分查找对比

| 特性 | 二分查找 | 插值查找 |
|------|---------|---------|
| 分割点 | 中间 | 按值估算 |
| 均匀数据 | O(log n) | O(log log n) |
| 最坏 | O(log n) | O(n) |
| 适用 | 通用有序数据 | 均匀分布数据 |
| 运算 | 除法/移位 | 乘除法 |

## 适用场景

- 数据均匀分布：如均匀 ID、均匀时间戳
- 数据量大且分布已知：可以比二分快很多

## 常见陷阱

1. **分布不均匀**：可能退化到 O(n)
2. **除零**：arr[high] === arr[low] 时需要特殊处理
3. **浮点精度**：pos 计算可能有精度问题
