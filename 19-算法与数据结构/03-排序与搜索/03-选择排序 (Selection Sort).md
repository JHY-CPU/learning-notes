## 选择排序 (Selection Sort)

  选择排序每次从未排序部分找到最小（或最大）元素，将其放到已排序部分的末尾。


>
    **复杂度分析：**时间复杂度 O(n^2)，空间 O(1)。不稳定排序。


  ## 算法步骤


    - 将数组分为已排序和未排序两部分

    - 在未排序部分中找到最小元素

    - 将最小元素与未排序部分的第一个元素交换

    - 已排序部分增加一个元素，重复 n-1 次



  ## 代码实现


```
function selectionSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < n; j++) {
      if (arr[j] < arr[minIdx]) minIdx = j;
    }
    if (minIdx !== i) {
      [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
    }
  }
  return arr;
}```

  ## 交互演示




  点击按钮开始演示
