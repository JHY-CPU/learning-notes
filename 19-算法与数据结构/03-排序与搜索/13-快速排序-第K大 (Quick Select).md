## 快速选择 (Quick Select)

  快速选择算法用于在未排序数组中查找第 K 大（或第 K 小）元素。它基于快速排序的分区思想，但只需递归处理一侧，平均时间复杂度为 O(n)。


>
    **与快速排序的区别：**快速排序需要对分区两侧递归排序，而快速选择只对包含目标元素的那一侧递归。


  ## 算法思路


    - 选择一个基准元素进行分区操作

    - 检查基准元素的位置 pivotIndex：


      - 如果 pivotIndex === k，则找到答案

      - 如果 pivotIndex > k，在左半部分继续查找

      - 如果 pivotIndex < k，在右半部分继续查找




  ## 代码实现


```
function quickSelect(arr, k) {
  // 查找第 k 小元素 (0-indexed)
  function select(left, right) {
    if (left === right) return arr[left];
    const pivotIndex = partition(arr, left, right);
    if (k === pivotIndex) return arr[k];
    if (k < pivotIndex) return select(left, pivotIndex - 1);
    return select(pivotIndex + 1, right);
  }
  return select(0, arr.length - 1);
}

// 第 K 大 = 第 (n - k) 小
function findKthLargest(arr, k) {
  return quickSelect(arr, arr.length - k);
}```

  ## 交互演示

  在随机数组中查找第 K 大元素：
