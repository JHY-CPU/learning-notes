# 前缀和专题 (Prefix Sum)

## 一、概念定义与原理

### 1.1 一维前缀和

前缀和数组 $S[i] = \sum_{j=0}^{i-1} a[j]$，其中 $S[0] = 0$。

区间和：$\sum_{j=l}^{r} a[j] = S[r+1] - S[l]$

### 1.2 二维前缀和

$$S[i][j] = a[i][j] + S[i-1][j] + S[i][j-1] - S[i-1][j-1]$$

子矩阵和：$S[x_2][y_2] - S[x_1-1][y_2] - S[x_2][y_1-1] + S[x_1-1][y_1-1]$

### 1.3 前缀和+哈希

统计和为 $k$ 的子数组个数：维护前缀和出现次数，对每个位置查询 `当前前缀和 - k` 的出现次数。

---

## 二、核心算法

### 2.1 构建与查询

- 构建：$O(n)$
- 查询：$O(1)$

### 2.2 前缀和+哈希

$$\text{count} = \sum_{i=0}^{n} \text{hash}[S[i] - k]$$

其中 $S[i]$ 是前缀和，$\text{hash}$ 记录每个前缀和出现的次数。

---

## 三、代码实现

### 3.1 一维前缀和 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 构建前缀和
vector<long long> build_prefix(vector<int>& a) {
    int n = a.size();
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i+1] = S[i] + a[i];
    return S;
}
// query(l, r) = S[r+1] - S[l]
```

### 3.2 二维前缀和 - C++

```cpp
vector<vector<long long>> build_prefix_2d(vector<vector<int>>& a) {
    int m = a.size(), n = a[0].size();
    vector<vector<long long>> S(m+1, vector<long long>(n+1, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            S[i+1][j+1] = a[i][j] + S[i][j+1] + S[i+1][j] - S[i][j];
    return S;
}

// 子矩阵 [r1,c1] 到 [r2,c2] 的和
long long query_2d(vector<vector<long long>>& S, int r1, int c1, int r2, int c2) {
    return S[r2+1][c2+1] - S[r1][c2+1] - S[r2+1][c1] + S[r1][c1];
}
```

### 3.3 和为K的子数组个数 - C++

```cpp
// LeetCode 560: 和为K的子数组
int subarray_sum(vector<int>& nums, int k) {
    unordered_map<long long, int> prefix_count;
    prefix_count[0] = 1;
    long long sum = 0;
    int result = 0;
    for (int x : nums) {
        sum += x;
        result += prefix_count[sum - k];
        prefix_count[sum]++;
    }
    return result;
}
```

### 3.4 Python 实现

```python
def build_prefix(a):
    S = [0]
    for x in a: S.append(S[-1] + x)
    return S

def query(S, l, r):
    return S[r+1] - S[l]

def subarray_sum(nums, k):
    prefix_count = {0: 1}; s = 0; result = 0
    for x in nums:
        s += x
        result += prefix_count.get(s - k, 0)
        prefix_count[s] = prefix_count.get(s, 0) + 1
    return result

def build_prefix_2d(a):
    m, n = len(a), len(a[0])
    S = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            S[i+1][j+1] = a[i][j] + S[i][j+1] + S[i+1][j] - S[i][j]
    return S

a = [1,2,3,4,5]
S = build_prefix(a)
print(query(S, 1, 3))  # 9 (2+3+4)
print(subarray_sum([1,1,1], 2))  # 2
```

### 3.5 异或前缀和

```cpp
// 异或前缀和：找异或和为k的子数组
int subarray_xor(vector<int>& nums, int k) {
    unordered_map<int, int> prefix_count;
    prefix_count[0] = 1;
    int xor_sum = 0, result = 0;
    for (int x : nums) {
        xor_sum ^= x;
        result += prefix_count[xor_sum ^ k];
        prefix_count[xor_sum]++;
    }
    return result;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 构建前缀和 | $O(n)$ | $O(n)$ |
| 区间查询 | $O(1)$ | - |
| 二维构建 | $O(mn)$ | $O(mn)$ |
| 和为K计数 | $O(n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 303：** 区域和检索 - 数组不可变
2. **LeetCode 560：** 和为K的子数组
3. **LeetCode 304：** 二维区域和检索
4. **LeetCode 1314：** 矩阵区域和
5. **异或前缀和：** LeetCode 1442
