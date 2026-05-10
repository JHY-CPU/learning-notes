# 算法思维训练 (Algorithm Mindset)

## 一、问题分析方法

### 1.1 问题抽象

将实际问题转化为数学/数据结构模型：

1. **识别数据结构：** 问题中的实体和关系对应什么结构？
   - 一对一关系 → 链表/数组
   - 一对多关系 → 树
   - 多对多关系 → 图
   - 键值映射 → 哈希表

2. **识别算法模式：** 问题要求什么？
   - 最优值 → DP/贪心
   - 所有方案 → 回溯/DFS
   - 最短路径 → BFS/Dijkstra
   - 查找 → 二分/哈希

### 1.2 降维思想

- 二维问题先思考一维解法
- 树的问题先思考链表解法
- 图的问题先思考树的解法

### 1.3 逆向思维

- 正向困难时考虑反向（如逆序处理）
- 直接计算困难时考虑补集
- 动态困难时考虑静态预处理

---

## 二、算法分类决策树

```
问题类型判断：
├── 求最优值/计数/可行性？
│   ├── 有重叠子问题 → 动态规划
│   ├── 局部最优=全局最优 → 贪心
│   └── 搜索空间有限 → 记忆化搜索
├── 求所有方案？
│   ├── 组合/排列/子集 → 回溯
│   └── 图上路径 → DFS
├── 求最短路径？
│   ├── 无权图 → BFS
│   ├── 非负权图 → Dijkstra
│   └── 有负权 → Bellman-Ford
├── 求连通性/分组？
│   ├── 集合合并 → 并查集
│   └── 遍历 → BFS/DFS
└── 有序数据查找？
    ├── 精确查找 → 二分查找
    └── 范围查询 → 二分+前缀和
```

---

## 三、解题思维模式

### 3.1 分治思维

**适用：** 问题可独立分解为子问题。

```python
# 最大子数组和 — 分治解法
def max_subarray_dc(nums):
    def solve(left, right):
        if left == right: return nums[left]
        mid = (left + right) // 2

        # 左半最大
        left_max = solve(left, mid)
        # 右半最大
        right_max = solve(mid+1, right)

        # 跨中点的最大
        left_cross = float('-inf')
        s = 0
        for i in range(mid, left-1, -1):
            s += nums[i]
            left_cross = max(left_cross, s)

        right_cross = float('-inf')
        s = 0
        for i in range(mid+1, right+1):
            s += nums[i]
            right_cross = max(right_cross, s)

        return max(left_max, right_max, left_cross + right_cross)

    return solve(0, len(nums)-1)
```

### 3.2 双指针思维

**适用：** 有序数组、链表、回文。

```python
# 盛最多水的容器 (LeetCode 11)
def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        h = min(height[left], height[right])
        best = max(best, h * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
```

### 3.3 单调栈思维

**适用：** 找下一个更大/更小元素。

```python
# 下一个更大元素
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

### 3.4 前缀和思维

**适用：** 区间和查询、子数组和。

```python
# 和为K的子数组个数
def subarray_sum_k(nums, k):
    count, prefix, seen = 0, 0, {0: 1}
    for num in nums:
        prefix += num
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
```

---

## 四、调试思维

### 4.1 小数据测试法

```python
def debug_test(func, test_cases):
    for i, (inputs, expected) in enumerate(test_cases):
        result = func(*inputs)
        if result != expected:
            print(f"Test {i} failed:")
            print(f"  Input: {inputs}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
```

### 4.2 手动模拟

对于复杂算法，先用小例子手动走一遍，验证思路正确。

### 4.3 不变式法

维护循环不变式，确保每步迭代后不变式成立。

---

## 五、优化思维

### 5.1 常见优化路径

```
暴力O(n!) → 剪枝 → 回溯O(2^n) → 记忆化 → DP O(n^k) → 优化 → O(n log n) 或 O(n)
```

### 5.2 空间换时间

- 预处理（排序、前缀和、哈希表）
- 缓存结果（记忆化、DP表）
- 辅助数据结构（堆、并查集、线段树）

### 5.3 排序预处理

很多问题排序后更容易解决：
- 双指针需要有序
- 二分查找需要有序
- 贪心策略通常需要排序
- 去重只需遍历相邻元素

---

## 六、面试思维框架

```
1. 听题 → 确认输入输出、数据范围、边界情况
2. 暴力 → 先说最简单的方法和复杂度
3. 观察 → 找瓶颈、找规律、找特殊性质
4. 优化 → 换数据结构、换算法、剪枝
5. 实现 → 先写框架，再处理边界
6. 验证 → 小数据、大数据、特殊数据
```
