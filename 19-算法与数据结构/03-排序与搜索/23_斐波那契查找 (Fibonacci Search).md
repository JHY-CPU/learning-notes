# 24-斐波那契查找 (Fibonacci Search)

斐波那契查找基于斐波那契数列进行分割，用黄金比例（约0.618）决定查找位置，只用加减法运算。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 平均时间 | O(log n) |
| 最坏时间 | O(log n) |
| 空间 | O(1) |

## JavaScript 实现

```javascript
function fibonacciSearch(arr, target) {
  const n = arr.length;

  // 找到 >= n 的最小斐波那契数
  let fibK2 = 0;  // F(k-2)
  let fibK1 = 1;  // F(k-1)
  let fibK = fibK2 + fibK1;  // F(k)

  while (fibK < n) {
    fibK2 = fibK1;
    fibK1 = fibK;
    fibK = fibK2 + fibK1;
  }

  let offset = -1;
  while (fibK > 1) {
    const i = Math.min(offset + fibK2, n - 1);

    if (arr[i] < target) {
      // 在右侧查找
      fibK = fibK1;
      fibK1 = fibK2;
      fibK2 = fibK - fibK1;
      offset = i;
    } else if (arr[i] > target) {
      // 在左侧查找
      fibK = fibK2;
      fibK1 = fibK1 - fibK2;
      fibK2 = fibK - fibK1;
    } else {
      return i;
    }
  }

  // 检查最后一个元素
  if (fibK1 === 1 && offset + 1 < n && arr[offset + 1] === target) {
    return offset + 1;
  }
  return -1;
}

// 测试
const arr = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100];
console.log(fibonacciSearch(arr, 85));  // 8
console.log(fibonacciSearch(arr, 33));  // -1
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int fibonacciSearch(vector<int>& arr, int target) {
    int n = arr.size();
    int fibK2 = 0, fibK1 = 1, fibK = 1;
    while (fibK < n) {
        fibK2 = fibK1;
        fibK1 = fibK;
        fibK = fibK2 + fibK1;
    }

    int offset = -1;
    while (fibK > 1) {
        int i = min(offset + fibK2, n - 1);
        if (arr[i] < target) {
            fibK = fibK1;
            fibK1 = fibK2;
            fibK2 = fibK - fibK1;
            offset = i;
        } else if (arr[i] > target) {
            fibK = fibK2;
            fibK1 = fibK1 - fibK2;
            fibK2 = fibK - fibK1;
        } else {
            return i;
        }
    }
    if (fibK1 == 1 && offset + 1 < n && arr[offset + 1] == target) return offset + 1;
    return -1;
}
```

## 与二分查找对比

| 特性 | 二分查找 | 斐波那契查找 |
|------|---------|-------------|
| 分割比例 | 1:1 | 1:1.618 |
| 运算 | 除法/移位 | 仅加减法 |
| 时间 | O(log n) | O(log n) |
| 适用 | 通用 | 嵌入式/无除法硬件 |

## 适用场景

- 嵌入式系统：没有除法指令的硬件
- 数据存储在外部介质：斐波那契分割减少磁盘访问
- 教学：理解黄金分割思想

## 常见陷阱

1. **初始化**：需要找到 >= n 的最小斐波那契数
2. **边界处理**：offset + fibK2 可能越界，用 min 限制
3. **最后检查**：fibK1 === 1 时需要额外检查一个元素
