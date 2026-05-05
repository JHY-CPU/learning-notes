## 冒泡排序 (Bubble Sort)

  冒泡排序重复遍历数组，依次比较相邻元素，如果顺序错误就交换它们。每轮遍历将最大（或最小）元素"冒泡"到末尾。


>
    **复杂度分析：**平均/最坏 O(n^2)，最优 O(n)。空间 O(1)。稳定排序。


  ## 算法步骤


    - 从第一个元素开始，依次比较相邻元素

    - 如果前一个比后一个大，则交换

    - 每轮结束后，最大的元素被放到末尾

    - 重复 n-1 轮，或某一轮没有交换时提前结束



  ## 代码实现


```
function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        swapped = true;
      }
    }
    if (!swapped) break; // 优化：提前结束
  }
  return arr;
}```

  ## 交互演示




  点击按钮开始演示
