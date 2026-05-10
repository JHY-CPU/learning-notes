# 位运算与状态压缩 (Bitmask DP)

## 一、状态压缩思想

### 1.1 核心概念

状态压缩DP用**整数的二进制位**表示集合中元素的选取状态。例如，`0b1011` 表示选了第0、1、3号元素。

**适用条件：** 元素数量 $n \leq 20$（$2^{20} \approx 10^6$）

### 1.2 位操作基础

```python
# 检查第i位是否为1（元素i是否被选中）
(mask >> i) & 1

# 将第i位设为1（选中元素i）
mask | (1 << i)

# 将第i位设为0（取消选中元素i）
mask & ~(1 << i)

# 枚举mask的所有非空子集
sub = mask
while sub:
    # 处理 sub
    sub = (sub - 1) & mask
```

---

## 二、经典问题

### 2.1 TSP旅行商问题

**问题：** 从城市0出发，访问所有城市恰好一次后回到0，求最短路径。

```python
def tsp(dist):
    n = len(dist)
    INF = float('inf')
    # dp[mask][i] = 从0出发，经过mask中所有城市，最后停在i的最短距离
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 从城市0出发

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)): continue
            if dp[mask][u] == INF: continue
            for v in range(n):
                if mask & (1 << v): continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    # 回到城市0
    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(n))
```

**C++ 实现：**

```cpp
int tsp(vector<vector<int>>& dist) {
    int n = dist.size();
    int INF = 1e9;
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask & (1 << u)) || dp[mask][u] == INF) continue;
            for (int v = 0; v < n; v++) {
                if (mask & (1 << v)) continue;
                int nm = mask | (1 << v);
                dp[nm][v] = min(dp[nm][v], dp[mask][u] + dist[u][v]);
            }
        }
    }

    int full = (1 << n) - 1, ans = INF;
    for (int i = 0; i < n; i++)
        ans = min(ans, dp[full][i] + dist[i][0]);
    return ans;
}
```

### 2.2 分配问题 (LeetCode 1655)

**问题：** 将n个任务分配给m个人，每个人完成所分配任务的最长时间最小化。

```python
def can_distribute(nums, quantity, m):
    from collections import Counter
    freq = list(Counter(nums).values())
    freq.sort(reverse=True)

    def can_do(person_idx, remaining):
        if person_idx == len(quantity):
            return True
        # 尝试将 quantity[person_idx] 分配给某个频率足够的值
        for i in range(len(freq)):
            if freq[i] >= remaining[person_idx]:
                freq[i] -= remaining[person_idx]
                if can_do(person_idx + 1, remaining):
                    return True
                freq[i] += remaining[person_idx]
        return False

    return can_do(0, quantity)
```

### 2.3 最短汉密顿路径

```python
def hamiltonian_path(dist):
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]

    # 从每个城市出发
    for i in range(n):
        dp[1 << i][i] = 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)): continue
            if dp[mask][u] == INF: continue
            for v in range(n):
                if mask & (1 << v): continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    full = (1 << n) - 1
    return min(dp[full])
```

### 2.4 最小代价工作分配

```python
def min_cost_assignment(cost):
    n = len(cost)
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        worker = bin(mask).count('1')  # 已分配的人数 = 下一个要分配的人
        if worker >= n: continue
        for j in range(n):
            if mask & (1 << j): continue  # 任务j已被分配
            new_mask = mask | (1 << j)
            dp[new_mask] = min(dp[new_mask], dp[mask] + cost[worker][j])

    return dp[(1 << n) - 1]
```

---

## 三、集合运算模板

```python
# 全集
full = (1 << n) - 1

# 补集
complement = full ^ mask

# 交集
intersection = mask1 & mask2

# 并集
union = mask1 | mask2

# 元素个数
count = bin(mask).count('1')
# C++: __builtin_popcount(mask)

# 枚举所有子集
sub = mask
while sub:
    # 处理 sub
    sub = (sub - 1) & mask
```

---

## 四、复杂度分析

| 问题 | 时间 | 空间 |
|------|------|------|
| TSP | $O(2^n \cdot n^2)$ | $O(2^n \cdot n)$ |
| 最短汉密顿 | $O(2^n \cdot n^2)$ | $O(2^n \cdot n)$ |
| 工作分配 | $O(2^n \cdot n)$ | $O(2^n)$ |
| 子集枚举 | $O(3^n)$ 总计 | — |

---

## 五、面试要点

1. **n的范围** — 状态压缩只适用于 $n \leq 20$
2. **位操作熟练度** — 必须熟练掌握基本操作
3. **空间优化** — 有时可以只用一维数组
4. **预处理** — 集合信息可以预处理加速
