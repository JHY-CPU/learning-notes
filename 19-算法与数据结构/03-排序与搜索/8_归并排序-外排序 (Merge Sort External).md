# 9-归并排序外排序 (External Merge Sort)

外排序适用于数据量过大、无法全部加载到内存的场景。归并排序由于顺序访问的特性，非常适合外排序。

## 两阶段流程

### 阶段一：生成初始归并段 (Runs)

- 将大文件分割为 <= 内存容量的小块
- 对每个小块使用内部排序（快速排序）
- 排序后的块写回磁盘作为归并段

### 阶段二：多路归并 (Multi-Way Merge)

- 同时打开 k 个归并段文件
- 使用最小堆从 k 个段中选择最小元素
- 将选出的元素写入输出文件

## JavaScript 实现

```javascript
// 模拟多路归并
class MinHeap {
  constructor() { this.data = []; }
  push(x) {
    this.data.push(x);
    this._up(this.data.length - 1);
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) { this.data[0] = last; this._down(0); }
    return top;
  }
  isEmpty() { return this.data.length === 0; }
  _up(i) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p].val <= this.data[i].val) break;
      [this.data[p], this.data[i]] = [this.data[i], this.data[p]];
      i = p;
    }
  }
  _down(i) {
    const n = this.data.length;
    while (true) {
      let s = i, l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.data[l].val < this.data[s].val) s = l;
      if (r < n && this.data[r].val < this.data[s].val) s = r;
      if (s === i) break;
      [this.data[s], this.data[i]] = [this.data[i], this.data[s]];
      i = s;
    }
  }
}

// k 路归并
function kWayMerge(segments) {
  const heap = new MinHeap();
  const result = [];

  // 每个段的第一个元素加入堆
  for (let i = 0; i < segments.length; i++) {
    if (segments[i].length > 0) {
      heap.push({ val: segments[i][0], segIdx: i, idx: 0 });
    }
  }

  while (!heap.isEmpty()) {
    const { val, segIdx, idx } = heap.pop();
    result.push(val);
    if (idx + 1 < segments[segIdx].length) {
      heap.push({ val: segments[segIdx][idx + 1], segIdx, idx: idx + 1 });
    }
  }
  return result;
}

// 测试
const segments = [[1, 5, 9], [2, 6, 10], [3, 7, 8], [4, 11, 12]];
console.log(kWayMerge(segments)); // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

## C++ 实现

```cpp
#include <vector>
#include <queue>
using namespace std;

struct Element {
    int val, segIdx, idx;
    bool operator>(const Element& o) const { return val > o.val; }
};

vector<int> kWayMerge(vector<vector<int>>& segments) {
    priority_queue<Element, vector<Element>, greater<Element>> pq;
    for (int i = 0; i < segments.size(); i++) {
        if (!segments[i].empty()) pq.push({segments[i][0], i, 0});
    }
    vector<int> result;
    while (!pq.empty()) {
        auto [val, si, idx] = pq.top(); pq.pop();
        result.push_back(val);
        if (idx + 1 < segments[si].size()) {
            pq.push({segments[si][idx + 1], si, idx + 1});
        }
    }
    return result;
}
```

## 优化策略

| 策略 | 效果 |
|------|------|
| 置换选择排序 | 生成比内存更大的初始归并段 |
| 双缓冲 | 隐藏 I/O 延迟 |
| 增加归并路数 k | 减少归并趟数 |
| 最佳归并树 | 类似哈夫曼树，最小化总 I/O |

## 复杂度

总 I/O 次数 = 2 * n * (1 + log_k(n/m))，其中 n 是总数据量，m 是内存容量，k 是归并路数。

## 常见陷阱

1. **k 路选择**：k 越大趟数越少，但每趟的比较开销越大
2. **内存分配**：需要为输入/输出缓冲区预留空间
3. **文件句柄**：打开太多文件可能超过系统限制
