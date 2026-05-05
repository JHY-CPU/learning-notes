## 归并排序求逆序对 (Count Inversions)

  逆序对是指数组中满足 `i < j` 且 `arr[i] > arr[j]` 的元素对。利用归并排序的合并过程，可以在 O(n log n) 时间内高效统计逆序对数量。


>
    **核心思想：**在合并两个有序子数组时，如果左子数组的元素 `arr[i] > arr[j]`（右子数组的当前元素），则说明左子数组中从 i 到 mid 的所有元素都与 arr[j] 构成逆序对。


  ## 算法原理

  假设合并时左子数组为 `[L_i, ..., L_m]`，右子数组为 `[R_j, ..., R_n]`：


    - 当 `L_i ≤ R_j` 时，正常合并，不产生逆序对

    - 当 `L_i > R_j` 时，则 `L_i, L_{i+1}, ..., L_m` 都与 `R_j` 构成逆序对，共 `mid - i + 1` 个



  ## 代码实现


```
let count = 0;

function mergeSortCount(arr, left, right) {
  if (left >= right) return;
  const mid = Math.floor((left + right) / 2);
  mergeSortCount(arr, left, mid);
  mergeSortCount(arr, mid + 1, right);
  merge(arr, left, mid, right);
}

function merge(arr, left, mid, right) {
  const temp = [];
  let i = left, j = mid + 1;
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp.push(arr[i++]);
    } else {
      count += mid - i + 1; // 统计逆序对
      temp.push(arr[j++]);
    }
  }
  while (i <= mid) temp.push(arr[i++]);
  while (j <= right) temp.push(arr[j++]);
  for (let k = 0; k < temp.length; k++) {
    arr[left + k] = temp[k];
  }
}```

  ## 交互演示
