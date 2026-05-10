# 代码调试技巧 (Debugging Tips)

## 一、常见错误类型

### 1.1 边界条件错误

**最常见的错误来源：**

```python
# 错误：忘记处理空输入
def find_max(nums):
    max_val = nums[0]  # IndexError if nums is empty
    for num in nums:
        max_val = max(max_val, num)
    return max_val

# 正确：先检查边界
def find_max(nums):
    if not nums:
        raise ValueError("Empty array")
    max_val = nums[0]
    for num in nums:
        max_val = max(max_val, num)
    return max_val
```

**高频边界用例：**
- 空数组/空字符串
- 单个元素
- 两个元素
- 全部相同
- 已排序（正序/逆序）
- 最大/最小值
- 负数/零

### 1.2 整数溢出

```cpp
// 错误：可能溢出
int mid = (left + right) / 2;

// 正确：防止溢出
int mid = left + (right - left) / 2;

// Python 无需担心，整数无限精度
```

### 1.3 死循环

```python
# 错误：left 永远不移动
while left <= right:
    mid = (left + right) // 2
    if nums[mid] < target:
        left = mid  # 应该是 mid + 1
    else:
        right = mid - 1

# 正确
while left <= right:
    mid = left + (right - left) // 2
    if nums[mid] < target:
        left = mid + 1
    elif nums[mid] > target:
        right = mid - 1
    else:
        return mid
```

### 1.4 浅拷贝 vs 深拷贝

```python
# 错误：引用同一对象
grid = [[0] * n] * m  # 所有行指向同一列表！

# 正确：独立对象
grid = [[0] * n for _ in range(m)]

# 回溯时的浅拷贝
result.append(path[:])  # 不是 result.append(path)
```

---

## 二、调试方法

### 2.1 打印调试法

```python
def debug_dp(dp):
    """打印DP表方便调试"""
    for row in dp:
        print(' '.join(f'{x:4}' for x in row))
    print()

# 在递归中打印
def dfs(node, depth=0):
    print("  " * depth + f"visiting {node.val}")
    # ...
    for child in node.children:
        dfs(child, depth + 1)
    print("  " * depth + f"leaving {node.val}")
```

### 2.2 断言法

```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        # 断言循环不变式
        assert 0 <= left <= mid <= right < len(nums)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 2.3 对拍法 (Stress Testing)

用暴力解法验证优化解法：

```python
import random

def stress_test(optimized_func, brute_func, n_tests=1000):
    for i in range(n_tests):
        # 生成随机输入
        nums = [random.randint(-100, 100) for _ in range(random.randint(0, 20))]
        target = random.randint(-200, 200)

        expected = brute_func(nums, target)
        result = optimized_func(nums, target)

        if result != expected:
            print(f"Test {i} FAILED!")
            print(f"  Input: {nums}, target={target}")
            print(f"  Expected: {expected}, Got: {result}")
            return

    print(f"All {n_tests} tests passed!")
```

### 2.4 二分定位法

当大数据出错时，将数据分成两半，分别测试，快速定位出错范围。

---

## 三、常见算法题调试技巧

### 3.1 DP调试

```python
def debug_knapsack(w, v, W):
    dp = [0] * (W + 1)
    for i in range(len(w)):
        print(f"考虑物品{i}: weight={w[i]}, value={v[i]}")
        for j in range(W, w[i] - 1, -1):
            old = dp[j]
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
            if dp[j] != old:
                print(f"  dp[{j}]: {old} -> {dp[j]}")
    return dp[W]
```

### 3.2 BFS调试

```python
def bfs_debug(graph, start):
    from collections import deque
    visited = {start}
    queue = deque([(start, 0)])
    print(f"BFS from {start}")

    while queue:
        node, dist = queue.popleft()
        print(f"  Visit {node}, distance={dist}")
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
                print(f"    Enqueue {neighbor}")
```

### 3.3 回溯调试

```python
def backtrack_debug(path, choices, depth=0):
    indent = "  " * depth
    print(f"{indent}backtrack(path={path})")

    if is_solution(path):
        print(f"{indent}  -> Found solution!")
        return

    for choice in choices:
        if is_valid(choice, path):
            path.append(choice)
            print(f"{indent}  choose {choice}")
            backtrack_debug(path, remaining(choices, choice), depth + 1)
            path.pop()
            print(f"{indent}  undo {choice}")
```

---

## 四、工具与技巧

### 4.1 Python调试器

```python
import pdb

def problematic_function():
    x = complex_calculation()
    pdb.set_trace()  # 断点
    y = another_calculation(x)
    return y

# 常用命令：
# n (next) — 下一行
# s (step) — 进入函数
# c (continue) — 继续到下一个断点
# p var — 打印变量
# l (list) — 显示当前代码
```

### 4.2 单元测试

```python
import unittest

class TestSolution(unittest.TestCase):
    def test_two_sum(self):
        self.assertEqual(two_sum([2,7,11,15], 9), [0,1])

    def test_empty(self):
        self.assertEqual(two_sum([], 0), [])

    def test_negative(self):
        self.assertEqual(two_sum([-3,4,3,90], 0), [0,2])

    def test_large(self):
        self.assertEqual(two_sum([1,2,3,4,5], 9), [3,4])
```

### 4.3 LeetCode测试用例技巧

**卡住时的调试策略：**
1. 手动模拟示例用例
2. 构造小数据测试
3. 检查边界条件
4. 对拍暴力与优化解法
