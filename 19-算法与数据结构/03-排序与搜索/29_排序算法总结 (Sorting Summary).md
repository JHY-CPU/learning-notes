# 30-排序算法总结 (Sorting Summary)

常见排序算法的全面总结。

## 速查表

| 算法 | 时间（平均/最坏） | 空间 | 稳定 | 核心思想 |
|------|-----------------|------|------|---------|
| 冒泡 | O(n²)/O(n²) | O(1) | 是 | 相邻比较交换 |
| 选择 | O(n²)/O(n²) | O(1) | 否 | 选最小放到前面 |
| 插入 | O(n²)/O(n²) | O(1) | 是 | 有序序列中插入 |
| 希尔 | O(n^1.3)/O(n²) | O(1) | 否 | 步长分组插入 |
| 归并 | O(n log n)/O(n log n) | O(n) | 是 | 分治+合并 |
| 快排 | O(n log n)/O(n²) | O(log n) | 否 | 分治+分区 |
| 堆排 | O(n log n)/O(n log n) | O(1) | 否 | 堆结构 |
| 计数 | O(n+k)/O(n+k) | O(k) | 是 | 统计频次 |
| 桶排 | O(n+k)/O(n²) | O(n) | 是 | 分桶+桶内排序 |
| 基数 | O(d(n+k))/O(d(n+k)) | O(n+k) | 是 | 按位排序 |

## JavaScript 实现

```javascript
// 排序算法选择器
function selectSortAlgorithm(n, opts = {}) {
  const { range, needStable, memoryLimited, nearlySorted } = opts;

  if (range && range <= 1e6) return { algo: '计数排序', time: 'O(n+k)' };
  if (memoryLimited) return { algo: '堆排序', time: 'O(n log n)', space: 'O(1)' };
  if (needStable) return { algo: '归并排序', time: 'O(n log n)' };
  if (nearlySorted) return { algo: '插入排序', time: 'O(n)' };
  if (n < 50) return { algo: '插入排序', time: 'O(n²)' };
  return { algo: '快速排序', time: 'O(n log n)' };
}

// 快速测试所有排序
function testAll() {
  const data = [38, 27, 43, 3, 9, 82, 10];
  console.log('原数组:', data);
  console.log('冒泡:', bubbleSort([...data]));
  console.log('选择:', selectionSort([...data]));
  console.log('插入:', insertionSort([...data]));
  console.log('归并:', mergeSort([...data]));
  console.log('快排:', quickSort([...data]));
  console.log('堆排:', heapSort([...data]));
  console.log('计数:', countingSort([...data]));
}
```

## 排序下界

比较排序的理论下界是 Omega(n log n)：
- n 个元素有 n! 种排列
- 决策树高度 >= log(n!) = Omega(n log n)
- 非比较排序可以突破这个下界

## 学习路径

1. 先掌握冒泡、选择、插入（理解基本思想）
2. 学习归并、快排（分治思想）
3. 理解堆排序（数据结构配合）
4. 掌握计数、基数、桶排序（非比较排序）
5. 学习优化技巧（三路快排、TimSort）

## 常见陷阱

1. **死记复杂度**：理解推导过程比记忆更重要
2. **忽视适用场景**：没有万能的排序算法
3. **忽视稳定性**：多键排序时稳定性关键
4. **过度优化**：工程中优先使用内置排序
