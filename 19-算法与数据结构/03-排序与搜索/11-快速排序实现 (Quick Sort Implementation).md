## 快速排序实现 (Quick Sort Implementation)

  本节展示快速排序的完整递归实现，包含 Hoare 分区方案和 Lomuto 分区方案两种版本。


>
    **注意：**递归深度在极端情况下可能达到 O(n)，可通过尾递归优化或改用迭代方式避免栈溢出。


  ## Lomuto 分区实现


```
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
}```

  ## Hoare 分区实现


```
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
}```

  ## 交互演示




  点击按钮开始演示
