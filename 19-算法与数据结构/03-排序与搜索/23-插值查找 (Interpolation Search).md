## 插值查找 (Interpolation Search)

  插值查找是二分查找的改进版，基于数据分布均匀的假设，通过公式预测目标值的位置，而不是简单取中间值。


>
    **核心公式：**pos = low + ((target - arr[low]) / (arr[high] - arr[low])) * (high - low)


  ## 工作原理

  想象在字典中查找"西瓜"：你不会从中间翻起，而是根据拼音"X"的位置大致翻到靠后的部分。插值查找就是这种思想的算法实现。


    - 适用于**有序**且**均匀分布**的数据

    - 平均时间复杂度 O(log log n)，比二分查找更快

    - 最坏情况 O(n)（数据分布极度不均匀时）



  ## 代码实现


```
function interpolationSearch(arr, target) {
  let low = 0, high = arr.length - 1;

  while (low <= high && target >= arr[low] && target <= arr[high]) {
    if (low === high) {
      return arr[low] === target ? low : -1;
    }

    // 插值公式
    const pos = low + Math.floor(
      ((target - arr[low]) / (arr[high] - arr[low])) * (high - low)
    );

    if (arr[pos] === target) return pos;
    if (arr[pos] < target) low = pos + 1;
    else high = pos - 1;
  }
  return -1;
}```

  ## 交互演示

  对比插值查找和二分查找的性能差异（使用均匀分布数据）：
