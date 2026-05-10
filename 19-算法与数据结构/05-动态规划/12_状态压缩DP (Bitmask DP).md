# 状态压缩DP (Bitmask DP)

## 1. 概念与定义

状态压缩DP（Bitmask DP）是利用**位运算**将集合状态压缩为一个整数的动态规划方法。当问题的状态可以用一个集合表示，且集合元素数量较少（通常不超过20个）时，可以用二进制数的每一位表示一个元素是否在集合中。

核心思想：
- 用一个 n 位二进制数表示集合：第 i 位为 1 表示元素 i 在集合中
- 状态转移通过**位运算**实现：`mask | (1<<i)` 添加元素，`mask ^ (1<<i)` 移除元素
- 时间复杂度：O(2^n * n)，适用于 n <= 20 的情况

典型应用：
- 旅行商问题（TSP）
- 最小路径覆盖
- 集合覆盖问题
- 状态选择问题

## 2. 状态定义与转移方程

### 2.1 旅行商问题（TSP）

```
dp[mask][i] = 已访问集合为mask、当前在城市i时的最小路径长度
dp[mask][i] = min(dp[mask ^ (1<<i)][j] + dist[j][i]) for j in mask\{i}
初始条件：dp[1<<i][i] = 0
答案：min(dp[(1<<n)-1][i]) for all i
```

### 2.2 最小代价分组

```
dp[mask] = 将集合mask中的元素分成若干组的最小代价
dp[mask] = min(dp[mask ^ sub] + cost[sub]) for sub in subsets of mask
dp[0] = 0
```

### 2.3 常用位运算

```python
# 检查第i位是否为1
(mask >> i) & 1

# 将第i位设为1（添加元素i）
mask | (1 << i)

# 将第i位设为0（移除元素i）
mask ^ (1 << i)  # 当第i位为1时
mask & ~(1 << i)

# 枚举mask的所有子集
sub = mask
while sub:
    # 处理子集sub
    sub = (sub - 1) & mask

# 统计1的个数
bin(mask).count('1')
# 或用内置函数
mask.bit_count()  # Python 3.10+
```

## 3. 算法实现

### 3.1 旅行商问题（TSP）

```python
def tsp(dist):
    """旅行商问题：访问所有城市恰好一次并返回起点的最短路径"""
    n = len(dist)
    INF = float('inf')

    # dp[mask][i]: 已访问城市集合为mask，当前在城市i的最小距离
    dp = [[INF] * n for _ in range(1 << n)]

    # 从每个城市出发
    for i in range(n):
        dp[1 << i][i] = 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1:
                continue
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    full = (1 << n) - 1
    return min(dp[full][i] for i in range(n))
```

### 3.2 分配问题（LeetCode 1655）

```python
def canDistribute(nums, quantities):
    """能否将m种商品分配给n个客户"""
    from collections import Counter
    count = list(Counter(nums).values())
    m = len(quantities)

    # 预处理：对于每种数量，可以满足哪些客户的子集
    n = len(count)
    full = (1 << m) - 1

    # dp[mask] = 是否可以用某些商品满足mask中的客户需求
    dp = [False] * (1 << m)
    dp[0] = True

    for c in count:
        # 计算当前商品能同时满足哪些客户需求的子集
        # 使用背包思想
        valid = set()
        valid.add(0)
        for i in range(m):
            if quantities[i] <= c:
                for s in list(valid):
                    valid.add(s | (1 << i))

        # 更新dp
        new_dp = dp[:]
        for mask in range(1 << m):
            if dp[mask]:
                for sub in valid:
                    new_dp[mask | sub] = True
        dp = new_dp

    return dp[full]
```

### 3.3 最短超串（LeetCode 5990）

```python
def shortestSuperstring(words):
    """最短公共超串"""
    n = len(words)
    # 预处理两两之间的重叠长度
    overlap = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(min(len(words[i]), len(words[j])), 0, -1):
                    if words[i][-k:] == words[j][:k]:
                        overlap[i][j] = k
                        break

    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    for i in range(n):
        dp[1 << i][i] = len(words[i])

    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] == INF:
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                new_mask = mask | (1 << v)
                new_len = dp[mask][u] + len(words[v]) - overlap[u][v]
                if new_len < dp[new_mask][v]:
                    dp[new_mask][v] = new_len
                    parent[new_mask][v] = u

    full = (1 << n) - 1
    last = min(range(n), key=lambda i: dp[full][i])

    # 还原路径
    path = []
    mask = full
    while last != -1:
        path.append(last)
        prev = parent[mask][last]
        mask ^= (1 << last)
        last = prev
    path.reverse()

    result = words[path[0]]
    for i in range(1, len(path)):
        ov = overlap[path[i-1]][path[i]]
        result += words[path[i]][ov:]
    return result
```

### 3.4 C++ 实现

```cpp
// TSP
int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    int INF = 1e9;
    vector<vector<int>> dp(1<<n, vector<int>(n, INF));
    for (int i = 0; i < n; i++) dp[1<<i][i] = 0;
    for (int mask = 0; mask < (1<<n); mask++)
        for (int u = 0; u < n; u++) {
            if (!(mask & (1<<u))) continue;
            for (int v = 0; v < n; v++) {
                if (mask & (1<<v)) continue;
                dp[mask|(1<<v)][v] = min(dp[mask|(1<<v)][v], dp[mask][u] + dist[u][v]);
            }
        }
    int full = (1<<n) - 1;
    return *min_element(dp[full].begin(), dp[full].end());
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| TSP | O(2^n * n²) | O(2^n * n) |
| 最小分组 | O(3^n) | O(2^n) |
| 集合覆盖 | O(3^n) | O(2^n) |

注意：枚举所有子集的子集的总复杂度为 O(3^n)。

## 5. 典型例题

### 例题1：公平分配饼干（LeetCode 2305）

```python
def distributeCookies(cookies, k):
    n = len(cookies)
    INF = float('inf')

    # 预处理每个子集的总和
    total = [0] * (1 << n)
    for mask in range(1 << n):
        for i in range(n):
            if (mask >> i) & 1:
                total[mask] += cookies[i]

    # dp[mask][j] = 前j个人分配了mask中饼干的最小不公平程度
    dp = [[INF] * (k + 1) for _ in range(1 << n)]
    dp[0][0] = 0

    for mask in range(1 << n):
        for j in range(k):
            if dp[mask][j] == INF:
                continue
            # 枚举mask的补集的子集
            remain = ((1 << n) - 1) ^ mask
            sub = remain
            while sub:
                new_mask = mask | sub
                dp[new_mask][j + 1] = min(dp[new_mask][j + 1], max(dp[mask][j], total[sub]))
                sub = (sub - 1) & remain

    return dp[(1 << n) - 1][k]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **n太大**：n > 25 时 2^n 太大，无法使用
2. **位运算错误**：优先级问题，记得加括号
3. **枚举子集遗漏**：空集也要考虑
4. **初始化错误**：不可达状态要设为 INF

### 6.2 优化技巧

1. **预处理**：提前计算所有子集的属性（如和、异或等）
2. **对称性剪枝**：某些问题中交换元素不改变答案
3. **记忆化搜索**：用字典代替数组，节省空间
