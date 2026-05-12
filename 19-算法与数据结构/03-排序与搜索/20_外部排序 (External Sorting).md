# 21-外部排序 (External Sorting)

外部排序用于处理无法一次性载入内存的大规模数据。核心策略是分治：先分块排序，再多路归并。

## 两阶段流程

### 阶段一：生成初始归并段 (Runs)
- 将大文件分割为 <= 内存容量的小块
- 对每个小块使用内部排序
- 排序后的块写回磁盘

### 阶段二：多路归并 (Multi-Way Merge)
- 同时打开 k 个归并段
- 使用最小堆选择最小元素写入输出

## JavaScript 实现

```javascript
// 模拟外部排序：分块 + 多路归并
class ExternalSort {
  constructor(memoryLimit) {
    this.memoryLimit = memoryLimit;
  }

  // 阶段一：分块排序
  createRuns(data) {
    const runs = [];
    for (let i = 0; i < data.length; i += this.memoryLimit) {
      const chunk = data.slice(i, i + this.memoryLimit);
      chunk.sort((a, b) => a - b);
      runs.push(chunk);
    }
    return runs;
  }

  // 阶段二：k 路归并
  kWayMerge(runs) {
    const heap = [];
    const result = [];

    // 初始化堆
    for (let i = 0; i < runs.length; i++) {
      if (runs[i].length > 0) {
        heap.push({ val: runs[i][0], runIdx: i, pos: 0 });
      }
    }
    heap.sort((a, b) => a.val - b.val);

    while (heap.length > 0) {
      const min = heap.shift(); // 取最小
      result.push(min.val);
      const nextPos = min.pos + 1;
      if (nextPos < runs[min.runIdx].length) {
        const next = { val: runs[min.runIdx][nextPos], runIdx: min.runIdx, pos: nextPos };
        // 插入保持有序
        let inserted = false;
        for (let i = 0; i < heap.length; i++) {
          if (next.val <= heap[i].val) { heap.splice(i, 0, next); inserted = true; break; }
        }
        if (!inserted) heap.push(next);
      }
    }
    return result;
  }

  sort(data) {
    const runs = this.createRuns(data);
    return this.kWayMerge(runs);
  }
}

// 测试
const data = [29, 25, 3, 49, 9, 37, 21, 43, 15, 33, 7, 11];
const sorter = new ExternalSort(4); // 内存限制为 4 个元素
console.log(sorter.sort(data)); // [3, 7, 9, 11, 15, 21, 25, 29, 33, 37, 43, 49]
```

## C++ 实现

```cpp
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

struct Elem { int val, run, pos; };
bool operator>(const Elem& a, const Elem& b) { return a.val > b.val; }

vector<int> kWayMerge(vector<vector<int>>& runs) {
    priority_queue<Elem, vector<Elem>, greater<Elem>> pq;
    for (int i = 0; i < runs.size(); i++)
        if (!runs[i].empty()) pq.push({runs[i][0], i, 0});

    vector<int> result;
    while (!pq.empty()) {
        auto [val, run, pos] = pq.top(); pq.pop();
        result.push_back(val);
        if (pos + 1 < runs[run].size()) pq.push({runs[run][pos+1], run, pos+1});
    }
    return result;
}
```

## 优化策略

| 策略 | 效果 |
|------|------|
| 置换选择排序 | 生成更大的初始归并段 |
| 双缓冲 | 并行 I/O 和计算 |
| 增加归并路数 k | 减少归并趟数 |
| 最佳归并树 | 最小化总 I/O |

## 复杂度

总 I/O = 2 * n * (1 + ceil(log_k(n/m)))，其中 n 为数据量，m 为内存，k 为归并路数。

## 常见陷阱

1. **k 路数选择**：k 越大趟数越少，但每趟比较开销越大
2. **文件句柄**：打开过多文件可能超系统限制
3. **缓冲区管理**：需要合理分配输入输出缓冲区
