## 顺序查找 (Linear Search)

  顺序查找是最简单的查找算法，依次遍历数组的每个元素，直到找到目标值或遍历结束。


>
    **复杂度分析：**平均 O(n)，最坏 O(n)，最好 O(1)。空间 O(1)。


  ## 算法特点


    - **优点：**实现简单，不需要数据有序，适用于任何数据结构

    - **缺点：**速度慢，大规模数据中效率低下

    - **适用场景：**小规模数据、无序数据、只查找一次的场景



  ## 代码实现


```
function linearSearch(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
  }
  return -1;
}

// 哨兵优化版（减少比较次数）
function linearSearchSentinel(arr, target) {
  const n = arr.length;
  const last = arr[n - 1];
  arr[n - 1] = target; // 将目标值放在末尾作为哨兵

  let i = 0;
  while (arr[i] !== target) i++;

  arr[n - 1] = last; // 恢复原值
  if (i < n - 1 || arr[n - 1] === target) return i;
  return -1;
}```

  ## 交互演示
