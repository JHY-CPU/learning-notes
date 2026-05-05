## 插入排序 (Insertion Sort)

  插入排序通过构建有序序列，对未排序数据在已排序序列中从后向前扫描，找到相应位置并插入。


>
    **复杂度分析：**平均/最坏 O(n^2)，最优 O(n)（已排序）。空间 O(1)。稳定排序。


  ## 算法步骤


    - 从第二个元素开始，视为要插入的元素

    - 从后向前扫描已排序部分

    - 如果已排序元素大于当前元素，将其后移一位

    - 找到合适位置后插入当前元素



  ## 代码实现


```
function insertionSort(arr) {
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    let key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
  return arr;
}```

  ## 交互演示




  点击按钮开始演示
