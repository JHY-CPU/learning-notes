# 32-单调队列 (Monotonic Queue)

单调队列是一种在双端队列基础上维护元素单调递增或递减的结构。主要用于在滑动窗口中高效查询最大值或最小值。

## 核心思想

- **单调递减队列**：队头始终是窗口内的最大值
- **单调递增队列**：队头始终是窗口内的最小值
- **存索引**：队列中存数组下标，便于判断窗口范围
- **均摊 O(1)**：每个元素最多入队出队各一次

## 滑动窗口最大值

```javascript
function maxSlidingWindow(nums, k) {
  const deque = []; // 存索引，对应值单调递减
  const result = [];

  for (let i = 0; i < nums.length; i++) {
    // 移除超出窗口的队头
    while (deque.length && deque[0] <= i - k) deque.shift();
    // 维护单调递减：移除队尾比当前小的元素
    while (deque.length && nums[deque[deque.length - 1]] < nums[i]) deque.pop();
    deque.push(i);
    // 窗口形成后记录结果
    if (i >= k - 1) result.push(nums[deque[0]]);
  }
  return result;
}

console.log(maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3));
// [3,3,5,5,6,7]
```

## C++ 实现

```cpp
#include <vector>
#include <deque>
using namespace std;

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq; // 存索引
    vector<int> result;
    for (int i = 0; i < nums.size(); i++) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && nums[dq.back()] < nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(nums[dq.front()]);
    }
    return result;
}

// 滑动窗口最小值（单调递增队列）
vector<int> minSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq;
    vector<int> result;
    for (int i = 0; i < nums.size(); i++) {
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        while (!dq.empty() && nums[dq.back()] > nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(nums[dq.front()]);
    }
    return result;
}
```

## 单调队列模板

```javascript
// 单调递减队列（求窗口最大值）
class MonotonicDecreasingQueue {
  constructor() { this.deque = []; }

  // 入队：移除队尾所有 <= val 的元素
  push(val) {
    while (this.deque.length && this.deque[this.deque.length - 1] < val) {
      this.deque.pop();
    }
    this.deque.push(val);
  }

  // 出队：只在队头等于 val 时移除
  pop(val) {
    if (this.deque.length && this.deque[0] === val) {
      this.deque.shift();
    }
  }

  // 获取最大值
  max() { return this.deque[0]; }
}
```

## 应用：接雨水（单调栈）

```javascript
function trap(height) {
  let stack = [], water = 0;
  for (let i = 0; i < height.length; i++) {
    while (stack.length && height[i] > height[stack[stack.length - 1]]) {
      const bottom = height[stack.pop()];
      if (!stack.length) break;
      const width = i - stack[stack.length - 1] - 1;
      const h = Math.min(height[i], height[stack[stack.length - 1]]) - bottom;
      water += width * h;
    }
    stack.push(i);
  }
  return water;
}
```

## 单调栈 vs 单调队列

| 结构 | 维护 | 操作 | 典型应用 |
|------|------|------|---------|
| 单调栈 | 栈内单调 | push/pop | 下一个更大元素、接雨水 |
| 单调队列 | 队列单调 | push/pop/shift | 滑动窗口最值 |

## 复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| 维护单调性 | O(1) 均摊 | - |
| 查询窗口最值 | O(1) | - |
| 滑动窗口最值整体 | O(n) | O(k) |

## 常见应用

- 滑动窗口最大/最小值
- 接雨水问题（单调栈）
- 股票价格移动窗口分析
- 实时数据流窗口统计
- 子数组最值问题

## 常见陷阱

1. **比较方向**：求最大值用递减队列，求最小值用递增队列
2. **窗口判断**：`i >= k - 1` 时才开始记录结果
3. **队头清理**：每次迭代都要检查队头是否超出窗口
4. **存值 vs 存索引**：存索引更方便判断窗口范围
