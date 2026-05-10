# 复杂度优化思路 (Complexity Optimization)

## 一、复杂度等级认知

### 1.1 数据规模与可行算法

| 数据规模 $n$ | 可接受复杂度 | 典型算法 |
|-------------|-------------|---------|
| $n \leq 10$ | $O(n!)$, $O(2^n)$ | 暴力枚举、回溯 |
| $n \leq 25$ | $O(2^n)$ | 状态压缩DP |
| $n \leq 500$ | $O(n^3)$ | Floyd、区间DP |
| $n \leq 5000$ | $O(n^2)$ | 普通DP、冒泡排序 |
| $n \leq 10^5$ | $O(n \log n)$ | 排序、线段树 |
| $n \leq 10^6$ | $O(n)$ | 线性扫描、哈希表 |
| $n \leq 10^8$ | $O(n)$ 勉强 | 需要常数优化 |
| $n > 10^8$ | $O(\sqrt{n})$, $O(\log n)$ | 数学、二分 |

---

## 二、常见优化路径

### 2.1 暴力到优化的典型升级

```
O(n!) 全排列暴力
  ↓ 加剪枝
O(2^n) 回溯+剪枝
  ↓ 用记忆化
O(n^k) 记忆化搜索
  ↓ 改递推
O(n^k) 动态规划
  ↓ 优化转移
O(n log n) 单调队列/线段树优化DP
```

### 2.2 两数之和的优化过程

```python
# 方法1: 暴力 O(n^2)
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# 方法2: 排序+双指针 O(n log n)
def two_sum_sort(nums, target):
    sorted_nums = sorted((x, i) for i, x in enumerate(nums))
    left, right = 0, len(nums) - 1
    while left < right:
        s = sorted_nums[left][0] + sorted_nums[right][0]
        if s == target:
            return [sorted_nums[left][1], sorted_nums[right][1]]
        elif s < target: left += 1
        else: right -= 1

# 方法3: 哈希表 O(n)
def two_sum_hash(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

---

## 三、各类型优化技巧

### 3.1 DP优化

**空间优化：**
```python
# 二维DP → 一维
# dp[i][j] → dp[j]（逆序遍历）

# 降维示例：01背包
dp = [0] * (W + 1)
for w, v in zip(weights, values):
    for j in range(W, w - 1, -1):
        dp[j] = max(dp[j], dp[j - w] + v)
```

**单调队列优化：**
```python
# 滑动窗口最大值 → 从 O(nk) 优化到 O(n)
from collections import deque

def max_window(nums, k):
    dq, result = deque(), []
    for i, x in enumerate(nums):
        while dq and dq[0] < i - k + 1: dq.popleft()
        while dq and nums[dq[-1]] < x: dq.pop()
        dq.append(i)
        if i >= k - 1: result.append(nums[dq[0]])
    return result
```

### 3.2 搜索优化

**双向BFS：**
```python
# 从起点和终点同时搜索，相遇时停止
def bidirectional_bfs(graph, start, end):
    if start == end: return 0
    front, back = {start}, {end}
    visited = {start, end}
    step = 0

    while front and back:
        step += 1
        # 扩展较小的一端
        if len(front) > len(back):
            front, back = back, front

        next_front = set()
        for node in front:
            for neighbor in graph[node]:
                if neighbor in back: return step
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_front.add(neighbor)
        front = next_front

    return -1
```

**迭代加深：**
```python
# 在深度限制内DFS，逐步增加深度限制
def iterative_deepening(start, max_depth):
    for depth in range(max_depth + 1):
        result = dfs_limited(start, depth, 0)
        if result is not None:
            return result
    return None
```

### 3.3 数据结构优化

```python
# 暴力区间求和 O(n) → 前缀和 O(1)
# 暴力区间修改 O(n) → 差分数组 O(1)
# 暴力查找 O(n) → 哈希表 O(1)
# 暴力排序 O(n^2) → 归并/快排 O(n log n)
# 暴力最近公共祖先 O(n) → 倍增 O(log n)
```

---

## 四、常数优化

当理论复杂度已经最优时，考虑降低常数因子。

### 4.1 位运算加速

```python
# 检查奇偶：n % 2 → n & 1
# 乘以2：n * 2 → n << 1
# 除以2：n // 2 → n >> 1
```

### 4.2 减少函数调用

```python
# 内联热点函数
# 使用局部变量而非全局变量
# 预计算常用值
```

### 4.3 内存局部性

```python
# 按行遍历二维数组（缓存友好）
for i in range(m):
    for j in range(n):
        process(matrix[i][j])
```

---

## 五、复杂度分析实战

### 5.1 分析步骤

1. **确定输入规模**
2. **计算操作次数上限**
3. **反推可行的复杂度**
4. **选择最优算法**

### 5.2 示例

```
n = 10^5, 1秒时限
→ 操作数 ≤ 10^8
→ O(n log n) ≈ 1.7 * 10^6 ✓
→ O(n^2) = 10^10 ✗
→ 结论：需要 O(n log n) 或更优
```

---

## 六、面试中的复杂度优化

**标准回答流程：**
1. "最直接的做法是暴力，时间 O(n^2)，空间 O(1)"
2. "但数据量到 10^5，n^2 会超时"
3. "如果我们用 [哈希表/排序/DP]，可以优化到 O(n)"
4. "具体做法是..."

**关键原则：**
- 先说暴力再优化，展示思维过程
- 用数据量推导复杂度要求
- 明确说出优化思路的关键点
