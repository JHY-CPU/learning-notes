# 栈与队列专题 (Stack & Queue Problems)

## 一、概念定义与原理

### 1.1 栈 (Stack)

**后进先出 (LIFO)** 的数据结构，只允许在栈顶进行插入和删除。

**核心操作：** `push`, `pop`, `top`, `empty` — 全部 $O(1)$

### 1.2 队列 (Queue)

**先进先出 (FIFO)** 的数据结构，一端入队，另一端出队。

**变种：**
- **双端队列 (Deque)：** 两端都可以插入和删除
- **优先队列 (Priority Queue)：** 按优先级出队，用堆实现
- **循环队列：** 固定大小，头尾相连

### 1.3 栈与队列的相互实现

- 两个栈实现队列：一个入栈，一个出栈
- 两个队列实现栈：每次 push 后将旧元素重新入队

---

## 二、核心技巧

### 2.1 单调栈

维护栈内元素的单调性，解决**下一个更大/更小元素**问题。

**模板：**
```python
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```

### 2.2 单调队列

维护队列的单调性，解决**滑动窗口最值**问题。

```python
from collections import deque

def max_sliding_window(nums, k):
    result, dq = [], deque()
    for i in range(len(nums)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

### 2.3 表达式求值

中缀转后缀：操作数直接输出，运算符按优先级入栈。
后缀求值：操作数入栈，遇运算符弹出两个计算。

---

## 三、经典题目详解

### 3.1 有效的括号 (LeetCode 20)

```python
def is_valid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in mapping:
            if not stack or stack[-1] != mapping[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return len(stack) == 0
```

### 3.2 最小栈 (LeetCode 155)

```python
class MinStack:
    def __init__(self):
        self.stack, self.min_stack = [], []

    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(
            min(val, self.min_stack[-1]) if self.min_stack else val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]
```

### 3.3 柱状图中最大矩形 (LeetCode 84)

```cpp
int largestRectangleArea(vector<int>& heights) {
    int n = heights.size();
    stack<int> stk;
    int maxArea = 0;
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        while (!stk.empty() && h < heights[stk.top()]) {
            int height = heights[stk.top()]; stk.pop();
            int width = stk.empty() ? i : i - stk.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        stk.push(i);
    }
    return maxArea;
}
```

### 3.4 每日温度 (LeetCode 739)

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result, stack = [0] * n, []
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)
    return result
```

### 3.5 字符串解码 (LeetCode 394)

```python
def decode_string(s):
    stack, curr_str, curr_num = [], "", 0
    for c in s:
        if c.isdigit():
            curr_num = curr_num * 10 + int(c)
        elif c == '[':
            stack.append((curr_str, curr_num))
            curr_str, curr_num = "", 0
        elif c == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += c
    return curr_str
```

### 3.6 数据流中的中位数 (LeetCode 295)

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # 大顶堆（存负数）
        self.hi = []  # 小顶堆

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 栈 push/pop | $O(1)$ | $O(n)$ |
| 单调栈(整体) | $O(n)$ | $O(n)$ |
| 单调队列(整体) | $O(n)$ | $O(k)$ |
| 优先队列插入 | $O(\log n)$ | $O(n)$ |
| 滑动窗口最大值 | $O(n)$ | $O(k)$ |

---

## 五、面试高频题

1. **LeetCode 20：** 有效的括号
2. **LeetCode 155：** 最小栈
3. **LeetCode 232：** 用栈实现队列
4. **LeetCode 84：** 柱状图中最大矩形
5. **LeetCode 739：** 每日温度
6. **LeetCode 394：** 字符串解码
7. **LeetCode 239：** 滑动窗口最大值
8. **LeetCode 295：** 数据流的中位数
9. **LeetCode 347：** 前K个高频元素
10. **LeetCode 42：** 接雨水（单调栈解法）
