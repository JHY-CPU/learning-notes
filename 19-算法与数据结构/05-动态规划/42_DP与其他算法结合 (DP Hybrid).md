# DP与其他算法结合 (DP Hybrid)

## 1. 概念与定义

动态规划经常需要与其他算法技巧结合使用，以实现更高效的解法。常见的结合方式包括：
- **DP + 二分查找**：优化转移过程
- **DP + 贪心**：混合策略
- **DP + 单调队列/栈**：滑动窗口优化
- **DP + 前缀和**：快速区间查询
- **DP + 图论**：DAG上的DP
- **DP + 树状数组/线段树**：高效维护最值

## 2. DP + 二分查找

### 2.1 最长递增子序列

```python
import bisect

def lengthOfLIS(nums):
    """O(nlogn) 贪心+二分"""
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

### 2.2 分割数组的最大值最小化

```python
def splitArray(nums, k):
    """二分答案 + DP/贪心验证"""
    def can_split(max_sum):
        count, curr = 1, 0
        for num in nums:
            if curr + num > max_sum:
                count += 1
                curr = num
                if count > k: return False
            else:
                curr += num
        return True

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

## 3. DP + 贪心

### 3.1 跳跃游戏（LeetCode 55）

```python
def canJump(nums):
    """贪心：维护能到达的最远位置"""
    max_reach = 0
    for i, num in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + num)
    return True

def minJumps(nums):
    """DP+贪心：最少跳跃次数"""
    jumps = 0
    cur_end = 0
    farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = farthest
    return jumps
```

### 3.2 加油站问题（LeetCode 134）

```python
def canCompleteCircuit(gas, cost):
    """贪心：找到起始位置"""
    if sum(gas) < sum(cost):
        return -1
    start = 0
    tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

## 4. DP + 前缀和

### 4.1 和为k的子数组（LeetCode 560）

```python
def subarraySum(nums, k):
    """前缀和 + 哈希表"""
    from collections import defaultdict
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    curr_sum = 0
    result = 0
    for num in nums:
        curr_sum += num
        result += prefix_count[curr_sum - k]
        prefix_count[curr_sum] += 1
    return result
```

### 4.2 二维前缀和

```python
def matrixBlockSum(mat, k):
    m, n = len(mat), len(mat[0])
    # 构建前缀和
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = mat[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]

    # 查询
    def query(r1, c1, r2, c2):
        r1, c1 = max(0, r1), max(0, c1)
        r2, c2 = min(m-1, r2), min(n-1, c2)
        return prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]

    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = query(i-k, j-k, i+k, j+k)
    return result
```

## 5. DP + 图论

### 5.1 DAG上的DP

```python
def longestPathInDAG(n, edges):
    """DAG上求最长路径"""
    from collections import defaultdict
    graph = defaultdict(list)
    in_degree = [0] * n
    for u, v, w in edges:
        graph[u].append((v, w))
        in_degree[v] += 1

    # 拓扑排序
    from collections import deque
    queue = deque(i for i in range(n) if in_degree[i] == 0)
    dp = [0] * n

    while queue:
        u = queue.popleft()
        for v, w in graph[u]:
            dp[v] = max(dp[v], dp[u] + w)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return max(dp)
```

### 5.2 计算DAG路径数

```python
def countPaths(n, edges):
    """DAG上计算路径数"""
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    dp = [0] * n
    dp[0] = 1  # 起点

    # BFS拓扑排序
    from collections import deque
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            dp[v] += dp[u]
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return dp[n-1]
```

## 6. DP + 单调栈

### 6.1 柱状图最大矩形（LeetCode 84）

```python
def largestRectangleArea(heights):
    stack = [-1]
    max_area = 0
    for i in range(len(heights)):
        while len(stack) > 1 and heights[stack[-1]] >= heights[i]:
            h = heights[stack.pop()]
            w = i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    while len(stack) > 1:
        h = heights[stack.pop()]
        w = len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    return max_area
```

## 7. DP + 树状数组

### 7.1 逆序对计数

```python
def reversePairs(nums):
    """树状数组求逆序对"""
    sorted_nums = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_nums)}

    n = len(sorted_nums)
    tree = [0] * (n + 1)

    def update(i, delta):
        while i <= n:
            tree[i] += delta
            i += i & (-i)

    def query(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    count = 0
    for num in reversed(nums):
        count += query(rank[num] - 1)
        update(rank[num], 1)

    return count
```

## 8. 总结

```
DP + 二分：优化搜索，如LIS O(nlogn)
DP + 贪心：混合策略，如跳跃游戏
DP + 前缀和：区间查询，如子数组和
DP + 图论：DAG上DP，拓扑排序
DP + 单调栈：柱状图问题
DP + 树状数组：逆序对、偏序关系
```
