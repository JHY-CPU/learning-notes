# 19-排序算法对比 (Sorting Comparison)

本节提供全面的排序算法对比，包含理论分析和实际选择指南。

## 理论对比

| 算法 | 平均 | 最坏 | 最好 | 空间 | 稳定 | 原地 |
|------|------|------|------|------|------|------|
| 冒泡 | O(n²) | O(n²) | O(n) | O(1) | 是 | 是 |
| 选择 | O(n²) | O(n²) | O(n²) | O(1) | 否 | 是 |
| 插入 | O(n²) | O(n²) | O(n) | O(1) | 是 | 是 |
| 希尔 | O(n^1.3) | O(n²) | O(n) | O(1) | 否 | 是 |
| 归并 | O(n log n) | O(n log n) | O(n log n) | O(n) | 是 | 否 |
| 快排 | O(n log n) | O(n²) | O(n log n) | O(log n) | 否 | 是 |
| 堆排 | O(n log n) | O(n log n) | O(n log n) | O(1) | 否 | 是 |
| 计数 | O(n+k) | O(n+k) | O(n+k) | O(k) | 是 | 否 |
| 桶排 | O(n+k) | O(n²) | O(n) | O(n) | 是 | 否 |
| 基数 | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | 是 | 否 |

## 实际性能对比（n=100000 随机整数）

| 算法 | 预计时间 | 说明 |
|------|---------|------|
| 快排 | ~15ms | 通常最快 |
| 归并 | ~20ms | 稳定 |
| 堆排 | ~30ms | O(1) 空间 |
| 计数 | ~5ms | 范围小时最快 |
| 插入 | ~3000ms | 不适用于大数据 |

## JavaScript 实现

```javascript
// 性能基准测试
function benchmark(sortFn, n, label) {
  const arr = Array.from({ length: n }, () => Math.floor(Math.random() * n));
  const start = performance.now();
  sortFn([...arr]);
  const elapsed = (performance.now() - start).toFixed(2);
  console.log(`${label}: ${elapsed}ms (n=${n})`);
}

const sorts = {
  'Bubble': bubbleSort,
  'Insertion': insertionSort,
  'Merge': mergeSort,
  'Quick': quickSort,
  'Heap': heapSort,
  'Counting': countingSort,
};

// 小规模测试
for (const [name, fn] of Object.entries(sorts)) {
  benchmark(fn, 10000, name);
}
```

## 选择决策树

```
需要排序？
├─ 数据范围很小（如0-100） → 计数排序
├─ 数据分布均匀 → 桶排序
├─ 整数且位数固定 → 基数排序
├─ 需要稳定？
│  ├─ 是 → 归并排序 / TimSort
│  └─ 否 → 快速排序 / 堆排序
├─ 内存非常有限？ → 堆排序
├─ 数据接近有序？ → 插入排序
└─ 大规模随机数据 → 快速排序
```

## 常见陷阱

1. **只看平均复杂度**：最坏情况在某些场景很重要
2. **忽视常数因子**：O(n log n) 不总是比 O(n²) 快
3. **忽视数据特征**：有序、重复、分布不均匀都会影响选择
4. **忽视稳定性**：多键排序时稳定性至关重要
