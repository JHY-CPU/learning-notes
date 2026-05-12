# 2-冒泡排序 (Bubble Sort)

冒泡排序重复遍历数组，依次比较相邻元素，如果顺序错误就交换它们。每轮遍历将最大元素"冒泡"到末尾。

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 最好（已排序） | O(n) | O(1) |
| 平均 | O(n²) | O(1) |
| 最坏（逆序） | O(n²) | O(1) |

稳定性：稳定。原地排序：是。

## JavaScript 实现

```javascript
// 基础版
function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;
      }
    }
    if (!swapped) break; // 优化：提前结束
  }
  return arr;
}

// 双向冒泡（鸡尾酒排序）
function cocktailSort(arr) {
  let l = 0, r = arr.length - 1;
  while (l < r) {
    let swapped = false;
    for (let i = l; i < r; i++) {
      if (arr[i] > arr[i + 1]) {
        [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]];
        swapped = true;
      }
    }
    r--;
    for (let i = r; i > l; i--) {
      if (arr[i] < arr[i - 1]) {
        [arr[i], arr[i - 1]] = [arr[i - 1], arr[i]];
        swapped = true;
      }
    }
    l++;
    if (!swapped) break;
  }
  return arr;
}

console.log(bubbleSort([5, 3, 8, 4, 2]));      // [2, 3, 4, 5, 8]
console.log(cocktailSort([5, 3, 8, 4, 2]));     // [2, 3, 4, 5, 8]
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}
```

## 算法过程

以 [5, 3, 8, 4, 2] 为例：
- 第1轮：[3, 5, 4, 2, 8] (8冒泡到末尾)
- 第2轮：[3, 4, 2, 5, 8] (5到位)
- 第3轮：[3, 2, 4, 5, 8] (4到位)
- 第4轮：[2, 3, 4, 5, 8] (3到位，完成)

## 适用场景

- 教学：原理最简单的排序算法
- 近乎有序的数据：带提前终止优化后接近 O(n)
- 小规模数据：n < 20 时可接受

## 变体

| 变体 | 特点 |
|------|------|
| 鸡尾酒排序 | 双向冒泡，对某些数据更快 |
| 带标志冒泡 | 记录最后一次交换位置，减少遍历范围 |
| 奇偶排序 | 并行友好的冒泡变体 |

## 常见陷阱

1. **忘记提前终止**：没有 swapped 标志导致总是 O(n²)
2. **循环范围错误**：内层循环到 n-1-i，不是 n-1
3. **性能期望**：即使优化后，平均仍是 O(n²)，不适合大规模数据
