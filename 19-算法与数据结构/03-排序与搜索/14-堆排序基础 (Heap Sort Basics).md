## 堆排序基础 (Heap Sort Basics)

  堆排序利用**堆**这种数据结构进行排序。堆是一棵完全二叉树，分为最大堆（父节点 ≥ 子节点）和最小堆（父节点 ≤ 子节点）。


>
    **复杂度分析：**时间复杂度 O(n log n)，空间 O(1)。不稳定排序。


  ## 堆的重要性质


    - 完全二叉树结构，可以用数组高效存储

    - 节点 i 的父节点下标为 `⌊(i-1)/2⌋`

    - 节点 i 的左子节点下标为 `2*i + 1`

    - 节点 i 的右子节点下标为 `2*i + 2`

    - 最后一个非叶子节点下标为 `⌊n/2⌋ - 1`



  ## 数组表示示例


```javascript
数组: [90, 70, 80, 50, 40, 30, 20, 10]

对应的完全二叉树:
          90
        /    \
      70      80
     /  \    /  \
    50  40  30  20
   /
  10```

  ## 堆的核心操作

  ### 下沉（Sift Down）

  当某个节点不满足堆性质时，将其与其较大的子节点交换，直到恢复堆性质。


```
function siftDown(arr, n, i) {
  let largest = i;
  const left = 2 * i + 1;
  const right = 2 * i + 2;
  if (left < n && arr[left] > arr[largest]) largest = left;
  if (right < n && arr[right] > arr[largest]) largest = right;
  if (largest !== i) {
    [arr[i], arr[largest]] = [arr[largest], arr[i]];
    siftDown(arr, n, largest);
  }
}```

  ## 建堆（Heapify）

  从最后一个非叶子节点开始，依次执行下沉操作，可在 O(n) 时间内将无序数组构建为堆。


```
function buildMaxHeap(arr) {
  const n = arr.length;
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    siftDown(arr, n, i);
  }
}```

  ## 交互演示：建堆过程
