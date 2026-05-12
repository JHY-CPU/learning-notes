# 22-顺序查找 (Linear Search)

顺序查找是最简单的查找算法，依次遍历每个元素直到找到目标或遍历结束。

## 复杂度分析

| 情况 | 时间 | 空间 |
|------|------|------|
| 最好（第一个） | O(1) | O(1) |
| 平均 | O(n) | O(1) |
| 最坏（最后一个或不存在） | O(n) | O(1) |

## JavaScript 实现

```javascript
// 基础版
function linearSearch(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) return i;
  }
  return -1;
}

// 哨兵优化版（减少比较次数）
function linearSearchSentinel(arr, target) {
  const n = arr.length;
  if (n === 0) return -1;
  const last = arr[n - 1];
  arr[n - 1] = target; // 设置哨兵

  let i = 0;
  while (arr[i] !== target) i++;

  arr[n - 1] = last; // 恢复
  if (i < n - 1 || arr[n - 1] === target) return i;
  return -1;
}

// 自组织查找：命中后前移（Move-to-Front）
class MoveToFrontSearch {
  constructor(arr) { this.arr = arr; }

  search(target) {
    for (let i = 0; i < this.arr.length; i++) {
      if (this.arr[i] === target) {
        // 命中后移到最前面
        if (i > 0) {
          [this.arr[0], this.arr[i]] = [this.arr[i], this.arr[0]];
        }
        return 0;
      }
    }
    return -1;
  }
}

// 转置法：命中后与前一个交换
function transposeSearch(arr, target) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === target) {
      if (i > 0) [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
      return i > 0 ? i - 1 : i;
    }
  }
  return -1;
}

console.log(linearSearch([10, 20, 30, 40, 50], 30)); // 2
console.log(linearSearchSentinel([10, 20, 30, 40, 50], 30)); // 2
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

int linearSearch(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

// 哨兵优化
int linearSearchSentinel(vector<int>& arr, int target) {
    int n = arr.size();
    if (n == 0) return -1;
    int last = arr[n - 1];
    arr[n - 1] = target;
    int i = 0;
    while (arr[i] != target) i++;
    arr[n - 1] = last;
    if (i < n - 1 || arr[n - 1] == target) return i;
    return -1;
}
```

## 自组织启发式

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| Move-to-Front | 命中移到最前 | 频繁访问热门元素 |
| 转置法 | 命中与前一个交换 | 渐进式调整 |
| 频率排序 | 按访问频率排列 | 已知访问分布 |

遵循 80/20 原则时，自组织线性查找可接近 O(1) 平均。

## 适用场景

- 无序数据，无法预排序
- 小规模数据（n < 20）
- 只查找一次，排序代价不值得
- 链表等不支持随机访问的结构

## 常见陷阱

1. **误用于大数据**：O(n) 在大数据下不可接受
2. **哨兵修改原数组**：需要恢复最后一个元素
3. **类型比较**：JS 中 `===` 严格比较，注意类型一致
