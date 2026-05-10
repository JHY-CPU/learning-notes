# 栈与队列专题 (Stack & Queue)

## 一、概念定义与原理

### 1.1 栈 (Stack)

**后进先出 (LIFO)** 的数据结构。支持 push、pop、top 操作。

### 1.2 队列 (Queue)

**先进先出 (FIFO)** 的数据结构。支持 push、pop、front 操作。

### 1.3 单调栈

栈内元素保持单调递增或递减。用于快速找到每个元素的**下一个更大/更小元素**。

### 1.4 单调队列

队列内元素保持单调性。用于维护**滑动窗口内的最值**。

---

## 二、核心算法

### 2.1 单调栈

**问题：** 对每个元素，找右侧第一个比它大的元素。

**方法：** 从右往左扫描，维护一个单调递减栈。栈顶即为答案。

### 2.2 单调队列

**问题：** 求长度为 $k$ 的滑动窗口内的最大值。

**方法：** 维护一个单调递减的双端队列。队首即为窗口最大值。

### 2.3 栈的应用

- 括号匹配
- 表达式求值
- 最小栈（O(1)获取最小值）

---

## 三、代码实现

### 3.1 单调栈 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 下一个更大元素
vector<int> next_greater(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> stk;
    for (int i = n - 1; i >= 0; i--) {
        while (!stk.empty() && nums[stk.top()] <= nums[i]) stk.pop();
        if (!stk.empty()) result[i] = nums[stk.top()];
        stk.push(i);
    }
    return result;
}

// 柱状图中最大矩形
int largest_rectangle(vector<int>& heights) {
    int n = heights.size(), result = 0;
    stack<int> stk;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (!stk.empty() && heights[stk.top()] > h) {
            int height = heights[stk.top()]; stk.pop();
            int width = stk.empty() ? i : i - stk.top() - 1;
            result = max(result, height * width);
        }
        stk.push(i);
    }
    return result;
}
```

### 3.2 单调队列 - C++

```cpp
// 滑动窗口最大值
vector<int> sliding_window_max(vector<int>& nums, int k) {
    deque<int> dq; // 存下标
    vector<int> result;
    for (int i = 0; i < nums.size(); i++) {
        // 移除超出窗口的元素
        while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
        // 维护单调递减
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k - 1) result.push_back(nums[dq.front()]);
    }
    return result;
}
```

### 3.3 最小栈 - C++

```cpp
class MinStack {
    stack<long long> stk;
    long long min_val;
public:
    void push(long long x) {
        if (stk.empty()) { stk.push(x); min_val = x; }
        else if (x >= min_val) { stk.push(x); }
        else { stk.push(2*x - min_val); min_val = x; }
    }
    void pop() {
        if (stk.top() < min_val) min_val = 2*min_val - stk.top();
        stk.pop();
    }
    long long top() {
        return stk.top() >= min_val ? stk.top() : min_val;
    }
    long long getMin() { return min_val; }
};
```

### 3.4 Python 实现

```python
from collections import deque

def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stk = []
    for i in range(n-1, -1, -1):
        while stk and nums[stk[-1]] <= nums[i]: stk.pop()
        if stk: result[i] = nums[stk[-1]]
        stk.append(i)
    return result

def sliding_window_max(nums, k):
    dq = deque(); result = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k: dq.popleft()
        while dq and nums[dq[-1]] <= x: dq.pop()
        dq.append(i)
        if i >= k - 1: result.append(nums[dq[0]])
    return result

print(next_greater([2,1,2,4,3]))          # [4,2,4,-1,-1]
print(sliding_window_max([1,3,-1,-3,5,3,6,7], 3))  # [3,3,5,5,6,7]
```

### 3.5 表达式求值

```cpp
// 中缀表达式求值
int evaluate(string expr) {
    stack<int> nums;
    stack<char> ops;
    int num = 0;
    for (int i = 0; i < expr.size(); i++) {
        char c = expr[i];
        if (isdigit(c)) { num = num * 10 + (c - '0'); }
        else {
            if (num) { nums.push(num); num = 0; }
            if (c == '(') ops.push(c);
            else if (c == ')') {
                while (ops.top() != '(') { /* 计算 */ ops.pop(); }
                ops.pop();
            } else {
                while (!ops.empty() && precedence(ops.top()) >= precedence(c))
                    { /* 计算 */ ops.pop(); }
                ops.push(c);
            }
        }
    }
    if (num) nums.push(num);
    while (!ops.empty()) { /* 计算 */ ops.pop(); }
    return nums.top();
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 单调栈 | $O(n)$ | $O(n)$ |
| 单调队列 | $O(n)$ | $O(k)$ |
| 最小栈 | $O(1)$ 各操作 | $O(n)$ |
| 表达式求值 | $O(n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 84：** 柱状图中最大矩形
2. **LeetCode 239：** 滑动窗口最大值
3. **LeetCode 155：** 最小栈
4. **下一个更大元素：** LeetCode 496/503
5. **接雨水：** LeetCode 42（单调栈经典应用）
