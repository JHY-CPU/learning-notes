# 回溯专题 (Backtracking)

## 一、概念定义与原理

### 1.1 回溯算法

回溯是一种通过**试探**和**回退**来搜索所有可能解的算法。本质是**DFS + 剪枝**。

### 1.2 回溯框架

```
void backtrack(路径, 选择列表):
    if 满足结束条件:
        收集结果
        return
    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

### 1.3 剪枝策略

- **可行性剪枝：** 当前状态已不满足约束，提前终止
- **最优性剪枝：** 当前解已不可能优于已知最优解
- **对称性剪枝：** 避免搜索等价的重复状态

---

## 二、经典问题

### 2.1 全排列

$n$ 个不同元素的全排列，共 $n!$ 种。

### 2.2 组合

从 $n$ 个元素中选 $k$ 个的组合，共 $C_n^k$ 种。

### 2.3 N皇后

在 $n \times n$ 棋盘上放 $n$ 个皇后，使得任意两个不在同一行、列、对角线。

### 2.4 子集枚举

枚举所有子集，共 $2^n$ 个。

---

## 三、代码实现

### 3.1 全排列 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

void permute(vector<int>& nums, vector<bool>& used,
             vector<int>& path, vector<vector<int>>& result) {
    if (path.size() == nums.size()) {
        result.push_back(path);
        return;
    }
    for (int i = 0; i < nums.size(); i++) {
        if (used[i]) continue;
        // 剪枝：去重（如果数组有重复元素）
        if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;
        used[i] = true;
        path.push_back(nums[i]);
        permute(nums, used, path, result);
        path.pop_back();
        used[i] = false;
    }
}
```

### 3.2 组合 - C++

```cpp
void combine(int n, int k, int start,
             vector<int>& path, vector<vector<int>>& result) {
    if (path.size() == k) {
        result.push_back(path);
        return;
    }
    // 剪枝：剩余元素不够
    for (int i = start; i <= n - (k - path.size()) + 1; i++) {
        path.push_back(i);
        combine(n, k, i + 1, path, result);
        path.pop_back();
    }
}
```

### 3.3 N皇后 - C++

```cpp
void solve_n_queens(int n, int row, vector<int>& queens,
                    vector<vector<string>>& result) {
    if (row == n) {
        vector<string> board(n, string(n, '.'));
        for (int i = 0; i < n; i++) board[i][queens[i]] = 'Q';
        result.push_back(board);
        return;
    }
    for (int col = 0; col < n; col++) {
        bool valid = true;
        for (int i = 0; i < row; i++) {
            if (queens[i] == col || abs(queens[i] - col) == abs(i - row)) {
                valid = false; break;
            }
        }
        if (!valid) continue;
        queens[row] = col;
        solve_n_queens(n, row + 1, queens, result);
    }
}
```

### 3.4 子集枚举 - C++

```cpp
// 方法1：回溯
void subsets(vector<int>& nums, int start,
             vector<int>& path, vector<vector<int>>& result) {
    result.push_back(path);
    for (int i = start; i < nums.size(); i++) {
        path.push_back(nums[i]);
        subsets(nums, i + 1, path, result);
        path.pop_back();
    }
}

// 方法2：位运算
vector<vector<int>> subsets_bit(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> result;
    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) subset.push_back(nums[i]);
        }
        result.push_back(subset);
    }
    return result;
}
```

### 3.5 Python 实现

```python
def permute(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:]); return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True; path.append(nums[i])
            backtrack(path, used)
            path.pop(); used[i] = False
    backtrack([], [False]*len(nums))
    return result

def combine(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k: result.append(path[:]); return
        for i in range(start, n - k + len(path) + 2):
            path.append(i)
            backtrack(i+1, path)
            path.pop()
    backtrack(1, [])
    return result

print(permute([1,2,3]))   # 6个排列
print(combine(4, 2))      # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 全排列 | $O(n!)$ | $O(n)$ 栈空间 |
| 组合 | $O(C_n^k)$ | $O(k)$ |
| N皇后 | $O(n!)$ | $O(n)$ |
| 子集 | $O(2^n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 46：** 全排列
2. **LeetCode 77：** 组合
3. **LeetCode 51：** N皇后
4. **LeetCode 78：** 子集
5. **LeetCode 39/40：** 组合总和（回溯+剪枝）
