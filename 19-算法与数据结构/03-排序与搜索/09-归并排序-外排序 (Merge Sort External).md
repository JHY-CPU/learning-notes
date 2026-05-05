## 归并排序外排序 (External Merge Sort)

  外排序（External Sorting）适用于数据量过大、无法全部加载到内存的场景。归并排序由于其顺序访问的特性，非常适合外排序。


>
    **应用场景：**数据库排序、大型文件排序、大数据处理等内存无法容纳全部数据的场景。


  ## 外排序过程

  ### 第一阶段：生成初始归并段（Run）

  将大文件分成多个小块，每块大小不超过可用内存。将每块数据读入内存，用快速排序等内部排序算法排序后写回磁盘。

  ### 第二阶段：多路归并

  将多个有序的归并段进行多路归并。使用优先队列（堆）从 k 个归并段中选取最小元素，逐步生成更大的有序段。

  ## 多路归并示意


```javascript
/* k 路归并使用最小堆 */
function kWayMerge(segments) {
  const heap = new MinHeap();
  const result = [];

  // 每个段取出第一个元素加入堆
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].length > 0) {
      heap.push({ val: segments[i][0], segIdx: i, idx: 0 });
    }
  }

  while (!heap.isEmpty()) {
    const { val, segIdx, idx } = heap.pop();
    result.push(val);
    // 从同一段取下一个元素
    if (idx + 1 < segments[segIdx].length) {
      heap.push({ val: segments[segIdx][idx + 1], segIdx, idx: idx + 1 });
    }
  }
  return result;
}```

  ## 交互演示：模拟外排序

  模拟将大数组分块排序后多路归并的过程。
