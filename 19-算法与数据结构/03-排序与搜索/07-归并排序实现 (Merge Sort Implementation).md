## 归并排序实现 (Merge Sort Implementation)

  归并排序的典型实现采用递归方式，包含两个核心函数：`mergeSort`（递归分治）和 `merge`（合并两个有序数组）。


>
    **实现要点：**需要 O(n) 的辅助空间来存储合并结果，但可以通过优化减少空间使用。


  ## 递归实现


```
function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  return merge(left, right);
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }
  return result.concat(left.slice(i)).concat(right.slice(j));
}```

  ## 原地归并（优化版）


```
function mergeSortInPlace(arr, left = 0, right = arr.length - 1) {
  if (left >= right) return;
  const mid = Math.floor((left + right) / 2);
  mergeSortInPlace(arr, left, mid);
  mergeSortInPlace(arr, mid + 1, right);
  mergeInPlace(arr, left, mid, right);
}

function mergeInPlace(arr, left, mid, right) {
  const temp = [];
  let i = left, j = mid + 1;
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) temp.push(arr[i++]);
    else temp.push(arr[j++]);
  }
  while (i <= mid) temp.push(arr[i++]);
  while (j <= right) temp.push(arr[j++]);
  for (let k = 0; k < temp.length; k++) {
    arr[left + k] = temp[k];
  }
}```

  ## 交互演示




  点击按钮开始演示
