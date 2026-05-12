# 11-快速排序实现 (Quick Sort Implementation)

本节展示快速排序的完整实现，包含 Lomuto 和 Hoare 两种分区方案，以及迭代版本。

## Lomuto 分区实现

```javascript
function quickSortLomuto(arr, low = 0, high = arr.length - 1) {
  if (low < high) {
    const pi = partitionLomuto(arr, low, high);
    quickSortLomuto(arr, low, pi - 1);
    quickSortLomuto(arr, pi + 1, high);
  }
  return arr;
}

function partitionLomuto(arr, low, high) {
  const pivot = arr[high];
  let i = low - 1;
  for (let j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
  return i + 1;
}
```

## Hoare 分区实现

```javascript
function quickSortHoare(arr, low = 0, high = arr.length - 1) {
  if (low < high) {
    const pi = partitionHoare(arr, low, high);
    quickSortHoare(arr, low, pi);
    quickSortHoare(arr, pi + 1, high);
  }
  return arr;
}

function partitionHoare(arr, low, high) {
  const pivot = arr[Math.floor((low + high) / 2)];
  let i = low - 1, j = high + 1;
  while (true) {
    do { i++; } while (arr[i] < pivot);
    do { j--; } while (arr[j] > pivot);
    if (i >= j) return j;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
```

## 迭代实现（用栈模拟递归）

```javascript
function quickSortIterative(arr) {
  const stack = [[0, arr.length - 1]];
  while (stack.length > 0) {
    const [low, high] = stack.pop();
    if (low < high) {
      const pi = partitionLomuto(arr, low, high);
      stack.push([low, pi - 1]);
      stack.push([pi + 1, high]);
    }
  }
  return arr;
}

console.log(quickSortLomuto([10, 7, 8, 9, 1, 5]));    // [1, 5, 7, 8, 9, 10]
console.log(quickSortHoare([10, 7, 8, 9, 1, 5]));      // [1, 5, 7, 8, 9, 10]
console.log(quickSortIterative([10, 7, 8, 9, 1, 5]));  // [1, 5, 7, 8, 9, 10]
```

## C++ 实现

```cpp
#include <vector>
#include <stack>
using namespace std;

// Lomuto
int partition(vector<int>& a, int l, int h) {
    int pivot = a[h], i = l - 1;
    for (int j = l; j < h; j++)
        if (a[j] <= pivot) swap(a[++i], a[j]);
    swap(a[i + 1], a[h]);
    return i + 1;
}

// Hoare
int partitionHoare(vector<int>& a, int l, int h) {
    int pivot = a[(l + h) / 2];
    int i = l - 1, j = h + 1;
    while (true) {
        do { i++; } while (a[i] < pivot);
        do { j--; } while (a[j] > pivot);
        if (i >= j) return j;
        swap(a[i], a[j]);
    }
}

// 迭代
void quickSortIter(vector<int>& a) {
    stack<pair<int,int>> st;
    st.push({0, (int)a.size() - 1});
    while (!st.empty()) {
        auto [l, h] = st.top(); st.pop();
        if (l < h) {
            int p = partition(a, l, h);
            st.push({l, p - 1});
            st.push({p + 1, h});
        }
    }
}
```

## 三种实现对比

| 实现 | 优点 | 缺点 |
|------|------|------|
| Lomuto 递归 | 简单易懂 | 交换多，递归栈 |
| Hoare 递归 | 交换少（3x） | 边界复杂 |
| 迭代 | 无栈溢出风险 | 需额外栈空间 |

## 常见陷阱

1. **Lomuto vs Hoare 返回值**：Lomuto 返回 pivot 位置，Hoare 返回分割点
2. **迭代栈顺序**：先推右区间再推左区间（LIFO）
3. **边界条件**：low < high 不是 low <= high
