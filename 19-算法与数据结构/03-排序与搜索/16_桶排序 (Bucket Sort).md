## 桶排序 (Bucket Sort)

  桶排序将数据分到有限数量的桶里，每个桶内分别排序，最后合并。适用于数据均匀分布的场景。


>
    **复杂度分析：**平均 O(n+k)，最坏 O(n^2)。空间 O(n)。稳定排序（取决于桶内排序算法）。


  ## 算法步骤


    - 确定桶的数量和范围

    - 将元素分配到对应的桶中

    - 对每个桶内的元素进行排序（通常使用插入排序）

    - 按顺序合并所有桶中的元素



  ## 代码实现


```
function bucketSort(arr, bucketSize = 5) {
  if (arr.length <= 1) return arr;
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const bucketCount = Math.floor((max - min) / bucketSize) + 1;
  const buckets = Array.from({length: bucketCount}, () => []);

  // 分配到桶
  for (const num of arr) {
    const idx = Math.floor((num - min) / bucketSize);
    buckets[idx].push(num);
  }

  // 每个桶内排序并合并
  const result = [];
  for (const bucket of buckets) {
    insertionSort(bucket);
    result.push(...bucket);
  }
  return result;
}

function insertionSort(arr) {
  for (let i = 1; i < arr.length; i++) {
    let key = arr[i], j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j]; j--;
    }
    arr[j + 1] = key;
  }
}```

  ## 交互演示
