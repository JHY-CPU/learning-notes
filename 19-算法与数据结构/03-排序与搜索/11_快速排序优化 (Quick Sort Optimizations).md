## 快速排序优化 (Quick Sort Optimizations)

  快速排序在面对特定数据（如已有序数组、大量重复元素）时可能退化到 O(n^2)。以下是几种经典优化策略。

  ## 优化1：三数取中法选基准

  避免选择到最小或最大元素作为基准。取左端、中间、右端三个元素的中值作为基准。


```
function medianOfThree(arr, low, high) {
  const mid = Math.floor((low + high) / 2);
  if (arr[low] > arr[mid]) [arr[low], arr[mid]] = [arr[mid], arr[low]];
  if (arr[low] > arr[high]) [arr[low], arr[high]] = [arr[high], arr[low]];
  if (arr[mid] > arr[high]) [arr[mid], arr[high]] = [arr[high], arr[mid]];
  // 将中值放到 high-1 位置
  [arr[mid], arr[high - 1]] = [arr[high - 1], arr[mid]];
  return arr[high - 1];
}```

  ## 优化2：三路快排（处理重复元素）

  当数组有大量重复元素时，将数组分为三部分：小于基准、等于基准、大于基准。


```
function quickSort3Way(arr, low = 0, high = arr.length - 1) {
  if (low >= high) return;
  const pivot = arr[low];
  let lt = low;   // arr[low...lt-1] < pivot
  let gt = high;  // arr[gt+1...high] > pivot
  let i = low + 1;
  while (i <= gt) {
    if (arr[i] < pivot) {
      [arr[lt], arr[i]] = [arr[i], arr[lt]];
      lt++; i++;
    } else if (arr[i] > pivot) {
      [arr[i], arr[gt]] = [arr[gt], arr[i]];
      gt--;
    } else {
      i++;
    }
  }
  quickSort3Way(arr, low, lt - 1);
  quickSort3Way(arr, gt + 1, high);
  return arr;
}```

  ## 优化3：小数组切换插入排序

  当子数组规模较小时（如 < 15），插入排序比快速排序更快。


```
const CUTOFF = 15;

function optimizedQuickSort(arr, low = 0, high = arr.length - 1) {
  if (high - low <= CUTOFF) {
    insertionSort(arr, low, high);
    return;
  }
  // ... 快速排序逻辑
  const pi = partition(arr, low, high);
  optimizedQuickSort(arr, low, pi - 1);
  optimizedQuickSort(arr, pi + 1, high);
}```

  ## 优化4：尾递归优化

  减少递归深度，避免栈溢出。


```
function tailRecursiveQuickSort(arr, low = 0, high = arr.length - 1) {
  while (low < high) {
    const pi = partition(arr, low, high);
    if (pi - low < high - pi) {
      tailRecursiveQuickSort(arr, low, pi - 1);
      low = pi + 1;
    } else {
      tailRecursiveQuickSort(arr, pi + 1, high);
      high = pi - 1;
    }
  }
}```

  ## 交互演示

  对比普通快速排序和三路快排处理重复元素的性能差异：
