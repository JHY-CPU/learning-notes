# 28-Top K 问题 (Top K Problem)

从数组中找出第 K 大（或前 K 大）的元素，是面试高频题目。

## 三种解法

| 方法 | 时间 | 空间 | 适用 |
|------|------|------|------|
| 排序 | O(n log n) | O(1) | 简单直接 |
| 快速选择 | O(n) 平均 | O(1) | 只需第K个 |
| 堆 | O(n log k) | O(k) | 前K个 |

## JavaScript 实现

```javascript
// 方法1：快速选择（平均 O(n)）
function findKthLargestQuickSelect(nums, k) {
  const target = nums.length - k;

  function select(left, right) {
    const pivot = nums[right];
    let i = left;
    for (let j = left; j < right; j++) {
      if (nums[j] <= pivot) {
        [nums[i], nums[j]] = [nums[j], nums[i]];
        i++;
      }
    }
    [nums[i], nums[right]] = [nums[right], nums[i]];

    if (i === target) return nums[i];
    if (i < target) return select(i + 1, right);
    return select(left, i - 1);
  }

  return select(0, nums.length - 1);
}

// 方法2：小顶堆（找前K大）
function findKthLargestHeap(nums, k) {
  // 维护大小为 k 的小顶堆
  const heap = new MinHeap();
  for (const n of nums) {
    heap.push(n);
    if (heap.size() > k) heap.pop();
  }
  return heap.peek();
}

class MinHeap {
  constructor() { this.data = []; }
  size() { return this.data.length; }
  peek() { return this.data[0]; }
  push(x) {
    this.data.push(x);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p] <= this.data[i]) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      while (true) {
        let s = i, l = 2*i+1, r = 2*i+2;
        if (l < this.data.length && this.data[l] < this.data[s]) s = l;
        if (r < this.data.length && this.data[r] < this.data[s]) s = r;
        if (s === i) break;
        [this.data[s], this.data[i]] = [this.data[i], this.data[s]];
        i = s;
      }
    }
    return top;
  }
}

// 方法3：排序
function findKthLargestSort(nums, k) {
  nums.sort((a, b) => a - b);
  return nums[nums.length - k];
}

// 前K大元素
function topK(nums, k) {
  const heap = new MinHeap();
  for (const n of nums) {
    heap.push(n);
    if (heap.size() > k) heap.pop();
  }
  const result = [];
  while (heap.size() > 0) result.push(heap.pop());
  return result.sort((a, b) => b - a); // 降序
}

// 测试
console.log(findKthLargestQuickSelect([3, 2, 1, 5, 6, 4], 2)); // 5
console.log(findKthLargestHeap([3, 2, 1, 5, 6, 4], 2));        // 5
console.log(topK([3, 2, 1, 5, 6, 4], 3));                       // [6, 5, 4]
```

## C++ 实现

```cpp
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

// 小顶堆
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int n : nums) {
        pq.push(n);
        if (pq.size() > k) pq.pop();
    }
    return pq.top();
}

// 快速选择
int quickSelect(vector<int>& a, int l, int r, int k) {
    int pivot = a[r], i = l;
    for (int j = l; j < r; j++)
        if (a[j] <= pivot) swap(a[i++], a[j]);
    swap(a[i], a[r]);
    if (i == k) return a[i];
    return i < k ? quickSelect(a, i+1, r, k) : quickSelect(a, l, i-1, k);
}
```

## 常见陷阱

1. **第K大 vs 第K小**：注意题目的具体含义
2. **堆类型**：找第K大用小顶堆，找第K小用大顶堆
3. **重复元素**：有重复时 "第K大" 的含义需要确认
4. **k 越界**：k > n 时需要特殊处理
