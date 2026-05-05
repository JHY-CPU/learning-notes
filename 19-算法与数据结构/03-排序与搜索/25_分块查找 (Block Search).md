## 分块查找 (Block Search)

  分块查找（也称索引顺序查找）将数据分成若干块，块内可以无序，但块间必须有序（第 i 块的最大值 < 第 i+1 块的最小值）。


>
    **复杂度分析：**O(log m + n/m)，其中 m 是块数。介于顺序查找和二分查找之间。


  ## 算法结构

  分块查找需要建立**索引表**，索引表存储每块的最大值和起始位置：


```
索引表结构：
[
  { max: 25, start: 0,  end: 4  },  // 块1: [15, 20, 25, 10, 22]
  { max: 48, start: 5,  end: 9  },  // 块2: [30, 35, 40, 48, 32]
  { max: 70, start: 10, end: 14 },  // 块3: [50, 60, 70, 55, 65]
]```

  ## 查找过程


    - 在索引表中二分查找目标值所在的块（找到 max ≥ target 的最小块）

    - 在对应块内顺序查找目标值



  ## 代码实现


```
function blockSearch(blocks, arr, target) {
  // 1. 在索引表中查找所属块
  let blockIdx = -1;
  for (let i = 0; i < blocks.length; i++) {
    if (target <= blocks[i].max) {
      blockIdx = i;
      break;
    }
  }
  if (blockIdx === -1) return -1;

  // 2. 在块内顺序查找
  const { start, end } = blocks[blockIdx];
  for (let i = start; i <= end; i++) {
    if (arr[i] === target) return i;
  }
  return -1;
}```

  ## 交互演示
