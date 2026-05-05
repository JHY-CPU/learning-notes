## 堆排序实现 (Heap Sort Implementation)

  堆排序分为两个阶段：建堆和排序。建堆后将堆顶元素（最大/最小值）与末尾元素交换，然后调整剩余元素为堆，重复此过程。


>
    **完整流程：**建堆（O(n)）→ 重复 n-1 次：交换堆顶和末尾 + 下沉调整（O(log n)）→ 总 O(n log n)


  ## 完整代码实现


```
function heapSort(arr) {
  const n = arr.length;

  // 建最大堆
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
    siftDown(arr, n, i);
  }

  // 逐个取出堆顶元素
  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]]; // 交换堆顶和末尾
    siftDown(arr, i, 0); // 调整剩余元素为堆
  }
  return arr;
}

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

  ## 交互演示




  点击按钮开始演示
