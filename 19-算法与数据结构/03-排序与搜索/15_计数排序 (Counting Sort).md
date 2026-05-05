## 计数排序 (Counting Sort)

  计数排序是一种**非比较排序**，通过统计每个元素出现的次数来确定位置。当数据范围有限时，效率极高。


>
    **复杂度分析：**时间复杂度 O(n+k)，空间 O(k)。其中 k 是数据范围。稳定排序。


  ## 算法步骤


    - 找出待排序数组的最大值和最小值，确定范围

    - 创建计数数组，统计每个元素出现的次数

    - 对计数数组进行前缀和累加，确定每个元素的最终位置

    - 从后向前遍历原数组，根据计数数组将元素放到正确位置



  ## 代码实现


```
function countingSort(arr) {
  if (arr.length <= 1) return arr;
  const max = Math.max(...arr);
  const min = Math.min(...arr);
  const range = max - min + 1;
  const count = new Array(range).fill(0);
  const output = new Array(arr.length);

  // 统计每个元素出现次数
  for (const num of arr) count[num - min]++;

  // 前缀和，确定位置
  for (let i = 1; i < range; i++) {
    count[i] += count[i - 1];
  }

  // 从后向前遍历，保证稳定性
  for (let i = arr.length - 1; i >= 0; i--) {
    const idx = arr[i] - min;
    output[count[idx] - 1] = arr[i];
    count[idx]--;
  }
  return output;
}```

  ## 交互演示
