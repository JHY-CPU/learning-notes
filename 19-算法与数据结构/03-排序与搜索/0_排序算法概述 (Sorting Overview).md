# 1-排序算法概述 (Sorting Overview)

排序算法是计算机科学中最基础的算法之一，用于将一组数据按特定顺序重新排列。

## 分类标准

| 分类维度 | 类型 | 说明 |
|---------|------|------|
| 比较 vs 非比较 | 比较排序 | 通过元素间比较决定顺序 |
| 比较 vs 非比较 | 非比较排序 | 利用元素特性（计数、基数） |
| 稳定性 | 稳定排序 | 保持相等元素的原始相对顺序 |
| 稳定性 | 不稳定排序 | 不保证相等元素的相对顺序 |
| 空间 | 原地排序 | 只需 O(1) 额外空间 |
| 空间 | 非原地排序 | 需要额外空间 |
| 数据位置 | 内部排序 | 数据全部在内存中 |
| 数据位置 | 外部排序 | 数据太大需磁盘辅助 |

## 排序算法全景对比

| 算法 | 平均 | 最坏 | 最好 | 空间 | 稳定 | 原地 |
|------|------|------|------|------|------|------|
| 冒泡排序 | O(n²) | O(n²) | O(n) | O(1) | 是 | 是 |
| 选择排序 | O(n²) | O(n²) | O(n²) | O(1) | 否 | 是 |
| 插入排序 | O(n²) | O(n²) | O(n) | O(1) | 是 | 是 |
| 希尔排序 | O(n^1.3) | O(n²) | O(n) | O(1) | 否 | 是 |
| 归并排序 | O(n log n) | O(n log n) | O(n log n) | O(n) | 是 | 否 |
| 快速排序 | O(n log n) | O(n²) | O(n log n) | O(log n) | 否 | 是 |
| 堆排序 | O(n log n) | O(n log n) | O(n log n) | O(1) | 否 | 是 |
| 计数排序 | O(n+k) | O(n+k) | O(n+k) | O(k) | 是 | 否 |
| 桶排序 | O(n+k) | O(n²) | O(n) | O(n) | 是 | 否 |
| 基数排序 | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | 是 | 否 |

## JavaScript 实现：排序算法框架

```javascript
// 排序算法选择指南
function chooseSortAlgorithm(n, range, needStable, memoryLimited) {
  if (range && range <= 1e6) return '计数排序 O(n+k)';
  if (memoryLimited) return '堆排序 O(n log n) O(1)空间';
  if (needStable) return '归并排序 O(n log n)';
  if (n < 50) return '插入排序 O(n²) 小数据常数小';
  return '快速排序 O(n log n) 通常最快';
}

// 测试排序算法稳定性
function testStability(sortFn) {
  const data = [
    { v: 3, id: 'a' }, { v: 1, id: 'b' },
    { v: 3, id: 'c' }, { v: 2, id: 'd' }
  ];
  sortFn(data, (a, b) => a.v - b.v);
  // 如果稳定，v=3 的 a 应该仍在 c 前面
  const idx3 = data.filter(x => x.v === 3).map(x => x.id);
  console.log(`稳定性: ${idx3[0] === 'a' ? '稳定' : '不稳定'}`);
}
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// STL sort: IntroSort (快排+堆排+插入排序混合)
// STL stable_sort: 归并排序变体

void demo() {
    vector<int> v = {5, 2, 8, 1, 9};
    sort(v.begin(), v.end());              // 不保证稳定
    stable_sort(v.begin(), v.end());       // 保证稳定
    partial_sort(v.begin(), v.begin() + 3, v.end()); // 只排前3个
    nth_element(v.begin(), v.begin() + 2, v.end());  // 找第3小
}
```

## 如何选择排序算法

- **小规模（n<100）**：插入排序，简单高效
- **中等规模**：快速排序通常最快
- **大规模**：归并排序或堆排序
- **数据范围有限**：计数排序、基数排序可达线性时间
- **需要稳定性**：归并排序、插入排序
- **内存受限**：堆排序 O(1) 空间

## 排序下界

比较排序的理论下界是 O(n log n)，因为 n 个元素有 n! 种排列，决策树高度至少 log(n!) = O(n log n)。非比较排序（计数、基数、桶排序）通过不比较元素直接确定位置，可以突破这个下界。

## 常见陷阱

1. **选择算法不看数据特征**：有序数据用快排会退化到 O(n²)
2. **忽视稳定性**：多键排序时稳定性很重要
3. **混用排序和搜索**：排序后的数组可以用二分查找，但排序本身有代价
4. **忽视常数因子**：O(n log n) 不代表一定比 O(n²) 快，小数据时插入排序更快
