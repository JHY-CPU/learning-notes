## 希尔排序 (Shell Sort)

  希尔排序是插入排序的改进版。通过将待排序数组按下标的一定增量分组，对每组使用插入排序，随着增量逐渐减少，每组元素越来越多，当增量减至1时，整个数组基本有序，最后进行一次插入排序。


>
    **复杂度分析：**时间复杂度取决于增量序列，平均 O(n^1.3)~O(n^2)。空间 O(1)。不稳定排序。


  ## 增量序列


    - **Shell 原始序列：**n/2, n/4, ..., 1

    - **Knuth 序列：**(3^k-1)/2 → ..., 121, 40, 13, 4, 1

    - **Sedgewick 序列：**..., 109, 41, 19, 5, 1



  ## 代码实现


```
function shellSort(arr) {
  const n = arr.length;
  for (let gap = Math.floor(n / 2); gap > 0; gap = Math.floor(gap / 2)) {
    for (let i = gap; i < n; i++) {
      let temp = arr[i];
      let j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
  return arr;
}```

  ## 交互演示




  点击按钮开始演示
