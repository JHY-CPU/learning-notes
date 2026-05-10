# 莫队算法 (Mo's Algorithm)

## 1. 概述

莫队算法（Mo's Algorithm）是一种**离线**处理区间查询的算法，由莫涛发明。它通过巧妙的查询排序，将所有查询的时间复杂度从 O(n * q) 降低到 O(n * sqrt(q)) 或 O((n+q) * sqrt(n))。

核心思想：**将查询按块排序，利用指针的移动来处理查询**。

## 2. 基本原理

### 2.1 问题模型

给定一个数组和若干区间查询 [l, r]，每个查询需要 O(1) 或 O(log n) 的时间来添加/移除一个元素。

### 2.2 算法思路

1. 将数组分为 sqrt(n) 个块
2. 将所有查询按 (l所在的块, r) 排序
3. 用两个指针维护当前区间 [L, R]，逐个处理查询
4. 每次移动指针，O(1) 更新答案

### 2.3 指针移动

```python
def process_query(l, r):
    """将当前区间调整到 [l, r]"""
    while L > l:
        L -= 1
        add(L)
    while R < r:
        R += 1
        add(R)
    while L < l:
        remove(L)
        L += 1
    while R > r:
        remove(R)
        R -= 1
```

## 3. 排序策略

### 3.1 普通莫队

```python
def mo_sort(queries, block_size):
    """莫队排序：按 (l所在块, r) 排序"""
    return sorted(queries, key=lambda q: (q[0] // block_size, q[1]))
```

### 3.2 奇偶优化

奇数块按 r 递减，偶数块按 r 递增，减少指针跳跃：

```python
def mo_sort_optimized(queries, block_size):
    """奇偶优化排序"""
    def sort_key(q):
        block = q[0] // block_size
        return (block, q[1] if block % 2 == 0 else -q[1])
    return sorted(queries, key=sort_key)
```

## 4. 完整实现（区间内不同元素个数）

### 4.1 经典例题：区间不同元素个数

```python
import math

def mo_algorithm(arr, queries):
    """
    莫队算法：求区间内不同元素的个数
    arr: 数组
    queries: [(l, r, query_id), ...]
    """
    n = len(arr)
    q = len(queries)
    block_size = max(1, int(n / math.sqrt(q)) if q > 0 else int(math.sqrt(n)))

    # 排序查询
    sorted_queries = sorted(queries, key=lambda x: (
        x[0] // block_size,
        x[1] if (x[0] // block_size) % 2 == 0 else -x[1]
    ))

    # 结果数组
    answers = [0] * q

    # 当前区间和计数
    L, R = 0, -1
    count = {}
    distinct = 0

    def add(idx):
        """添加 arr[idx] 到当前区间"""
        nonlocal distinct
        val = arr[idx]
        if val not in count:
            count[val] = 0
        if count[val] == 0:
            distinct += 1
        count[val] += 1

    def remove(idx):
        """从当前区间移除 arr[idx]"""
        nonlocal distinct
        val = arr[idx]
        count[val] -= 1
        if count[val] == 0:
            distinct -= 1

    # 处理每个查询
    for l, r, qid in sorted_queries:
        while L > l:
            L -= 1
            add(L)
        while R < r:
            R += 1
            add(R)
        while L < l:
            remove(L)
            L += 1
        while R > r:
            remove(R)
            R -= 1

        answers[qid] = distinct

    return answers


# 使用示例
if __name__ == "__main__":
    arr = [1, 2, 1, 3, 2, 1, 4, 3]
    queries = [(0, 4, 0), (1, 3, 1), (2, 7, 2), (0, 7, 3)]
    answers = mo_algorithm(arr, queries)
    for i, ans in enumerate(answers):
        print(f"查询 {queries[i][:2]}: {ans} 个不同元素")
```

## 5. 带修改的莫队（修改莫队）

### 5.1 原理

将查询和修改混合处理。每个查询关联一个"时间戳"，表示在该查询之前有多少次修改。

排序关键字变为 (l所在块, r所在块, 时间戳)。

### 5.2 实现

```python
def mo_with_updates(arr, queries):
    """
    带修改的莫队
    queries: [(l, r, time, query_id), ...]
    modifications: [(idx, old_val, new_val), ...]
    """
    n = len(arr)
    block_size = int(n ** (2/3))  # 三维分块

    # 按 (l块, r块, 时间) 排序
    def sort_key(q):
        return (q[0] // block_size,
                q[1] // block_size,
                q[2])

    queries.sort(key=sort_key)
    # ... 处理逻辑类似普通莫队，多了一个时间维度的移动
```

## 6. 回滚莫队（不删除莫队）

### 6.1 适用场景

当 add 操作容易实现但 remove 操作困难时使用。

### 6.2 核心思想

- 整块：每次新块开始时重置状态
- 对每个查询：先添加整块，再暴力添加散块，然后回滚散块的修改

```python
def rollback_mo(arr, queries):
    """回滚莫队"""
    n = len(arr)
    block_size = int(math.sqrt(n))

    # 按 (l所在块, r) 排序
    queries.sort(key=lambda q: (q[0] // block_size, q[1]))

    # ... 实现时 add 操作记录修改栈，查询后回滚
```

## 7. 时间复杂度分析

### 7.1 普通莫队

- 块大小：sqrt(n)
- 左指针移动：O(q * sqrt(n))
- 右指针移动：O(n * sqrt(n))
- 总复杂度：O((n + q) * sqrt(n))

### 7.2 带修改莫队

- 块大小：n^(2/3)
- 总复杂度：O(n^(5/3))

### 7.3 复杂度对比

| 变种 | 块大小 | 时间复杂度 |
|------|--------|-----------|
| 普通莫队 | sqrt(n) | O((n+q)*sqrt(n)) |
| 奇偶优化 | sqrt(n) | 略优 |
| 带修改 | n^(2/3) | O(n^(5/3)) |
| 回滚莫队 | sqrt(n) | O((n+q)*sqrt(n)) |

## 8. C++ 实现

```cpp
#include <algorithm>
#include <cmath>
using namespace std;

const int MAXN = 100005;
int arr[MAXN], cnt[MAXN], answers[MAXN];
int n, blockSize, curAns;

struct Query {
    int l, r, id;
};

bool cmp(const Query& a, const Query& b) {
    int ba = a.l / blockSize, bb = b.l / blockSize;
    if (ba != bb) return ba < bb;
    return (ba % 2 == 0) ? a.r < b.r : a.r > b.r;
}

void add(int idx) {
    cnt[arr[idx]]++;
    if (cnt[arr[idx]] == 1) curAns++;
}

void remove(int idx) {
    cnt[arr[idx]]--;
    if (cnt[arr[idx]] == 0) curAns--;
}

void mo(vector<Query>& queries) {
    blockSize = sqrt(n);
    sort(queries.begin(), queries.end(), cmp);

    int L = 0, R = -1;
    curAns = 0;

    for (auto& q : queries) {
        while (L > q.l) add(--L);
        while (R < q.r) add(++R);
        while (L < q.l) remove(L++);
        while (R > q.r) remove(R--);
        answers[q.id] = curAns;
    }
}
```

## 9. 使用示例

```python
if __name__ == "__main__":
    # 区间和查询
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = len(arr)

    def mo_sum(arr, queries):
        block_size = int(math.sqrt(n))

        def add(idx, state):
            state[0] += arr[idx]

        def remove(idx, state):
            state[0] -= arr[idx]

        queries = sorted(enumerate(queries),
                        key=lambda x: (x[1][0] // block_size, x[1][1]))

        answers = [0] * len(queries)
        L, R = 0, -1
        state = [0]  # 当前和

        for qid, (l, r) in queries:
            while L > l:
                L -= 1
                add(L, state)
            while R < r:
                R += 1
                add(R, state)
            while L < l:
                remove(L, state)
                L += 1
            while R > r:
                remove(R, state)
                R -= 1
            answers[qid] = state[0]

        return answers

    queries = [(0, 4), (2, 7), (1, 3), (5, 9)]
    print("区间和:", mo_sum(arr, queries))
```

## 10. 应用场景

1. 区间不同元素个数
2. 区间和、区间最大值
3. 区间众数
4. 区间逆序对
5. 带修改的区间查询

## 11. 注意事项

1. **必须离线**：所有查询需要预先知道
2. **块大小优化**：根据 n 和 q 的比例调整
3. **add/remove 可逆性**：普通莫队要求两者都容易实现
4. **内存限制**：回滚莫队需要记录修改栈

## 12. 总结

莫队算法是一种优雅的离线区间查询算法：
- 将查询排序后用指针移动处理
- 时间复杂度 O((n+q) * sqrt(n))
- 有多种变种适应不同场景
- 实现简单，是竞赛中的利器
