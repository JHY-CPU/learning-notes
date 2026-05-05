## 快速排序基础 (Quick Sort Basics)

  快速排序由 Tony Hoare 在 1959 年提出，是目前应用最广泛的排序算法之一。它同样采用**分治策略**，但不同于归并排序，快速排序是**原地排序**。


>
    **核心思想：**选择一个基准元素（pivot），将数组分为两部分：小于基准的放左边，大于基准的放右边。然后递归地对左右两部分排序。


  ## 算法步骤


    - **选择基准：**从数组中选择一个元素作为 pivot

    - **分区（Partition）：**重新排列数组，所有小于 pivot 的元素放在它左边，大于的放在右边

    - **递归排序：**对左右两个子数组递归应用快速排序



  ## 分区过程详解

  最经典的分区方式是 Lomuto 分区方案和 Hoare 分区方案：


```
// Lomuto 分区方案
function partition(arr, low, high) {
  const pivot = arr[high]; // 选最后一个元素为基准
  let i = low - 1;
  for (let j = low; j < high; j++) {
    if (arr[j] <= pivot) {
      i++;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
  [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
  return i + 1; // 返回 pivot 最终位置
}```

  ## 交互演示：分区过程

  点击按钮查看一次分区操作的详细过程。
