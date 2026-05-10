# 回溯算法专题 (Backtracking Problems)

## 一、概念定义与原理

### 1.1 回溯思想

回溯是一种**暴力搜索**策略，通过**剪枝**减少搜索空间。核心思想：

1. **选择：** 在当前决策点做出一个选择
2. **递归：** 基于这个选择继续搜索
3. **撤销：** 回退选择，尝试其他可能（回溯）

### 1.2 回溯模板

```python
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])  # 注意拷贝
        return

    for choice in choices:
        if not is_valid(choice, path):
            continue  # 剪枝
        path.append(choice)         # 做选择
        backtrack(path, next_choices)
        path.pop()                  # 撤销选择
```

### 1.3 回溯 vs 动态规划

| 特征 | 回溯 | 动态规划 |
|------|------|---------|
| 目标 | 找所有方案/任一方案 | 最优值/计数 |
| 方式 | 深度优先搜索 | 递推填表 |
| 剪枝 | 需要 | 无 |
| 重叠子问题 | 不利用 | 利用 |

---

## 二、经典题型

### 2.1 全排列 (LeetCode 46)

```python
def permute(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result
```

**C++ 实现：**

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> path;
    vector<bool> used(nums.size(), false);

    function<void()> backtrack = [&]() {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) continue;
            used[i] = true;
            path.push_back(nums[i]);
            backtrack();
            path.pop_back();
            used[i] = false;
        }
    };
    backtrack();
    return result;
}
```

### 2.2 组合总和 (LeetCode 39)

```python
def combination_sum(candidates, target):
    result = []
    candidates.sort()

    def backtrack(start, path, remain):
        if remain == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain: break  # 剪枝
            path.append(candidates[i])
            backtrack(i, path, remain - candidates[i])  # 可重复选
            path.pop()

    backtrack(0, [], target)
    return result
```

### 2.3 N皇后 (LeetCode 51)

```python
def solve_n_queens(n):
    result = []

    def backtrack(row, cols, diag1, diag2, board):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            board[row][col] = 'Q'
            backtrack(row+1, cols|{col}, diag1|{row-col},
                     diag2|{row+col}, board)
            board[row][col] = '.'

    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0, set(), set(), set(), board)
    return result
```

### 2.4 数独求解 (LeetCode 37)

```python
def solve_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty = []

    for i in range(9):
        for j in range(9):
            if board[i][j] == '.':
                empty.append((i, j))
            else:
                v = board[i][j]
                rows[i].add(v); cols[j].add(v)
                boxes[i//3*3+j//3].add(v)

    def backtrack(idx):
        if idx == len(empty): return True
        r, c = empty[idx]
        b = r//3*3+c//3
        for ch in '123456789':
            if ch in rows[r] or ch in cols[c] or ch in boxes[b]:
                continue
            board[r][c] = ch
            rows[r].add(ch); cols[c].add(ch); boxes[b].add(ch)
            if backtrack(idx+1): return True
            board[r][c] = '.'
            rows[r].remove(ch); cols[c].remove(ch); boxes[b].remove(ch)
        return False

    backtrack(0)
```

### 2.5 子集 (LeetCode 78)

```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### 2.6 单词搜索 (LeetCode 79)

```python
def exist(board, word):
    m, n = len(board), len(board[0])

    def dfs(i, j, k):
        if k == len(word): return True
        if (i < 0 or i >= m or j < 0 or j >= n
            or board[i][j] != word[k]):
            return False
        tmp = board[i][j]
        board[i][j] = '#'
        found = (dfs(i+1,j,k+1) or dfs(i-1,j,k+1)
              or dfs(i,j+1,k+1) or dfs(i,j-1,k+1))
        board[i][j] = tmp
        return found

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0): return True
    return False
```

---

## 三、剪枝技巧

### 3.1 常用剪枝策略

1. **排序剪枝：** 排序后，当前元素不满足条件时后续元素也不满足
2. **去重剪枝：** 同层跳过重复元素
3. **可行性剪枝：** 提前判断剩余值能否达到目标
4. **最优性剪枝：** 当前方案已不可能优于已知最优

```python
# 排列去重 (LeetCode 47)
def permute_unique(nums):
    result = []
    nums.sort()

    def backtrack(path, counter):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for x in counter:
            if counter[x] > 0:
                counter[x] -= 1
                path.append(x)
                backtrack(path, counter)
                path.pop()
                counter[x] += 1

    from collections import Counter
    backtrack([], Counter(nums))
    return result
```

---

## 四、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 全排列 | $O(n!)$ | $O(n)$ |
| 组合 | $O(C(n,k))$ | $O(k)$ |
| 子集 | $O(2^n)$ | $O(n)$ |
| N皇后 | $O(n!)$ | $O(n)$ |
| 数独 | $O(9^{empty})$ | $O(1)$ |

---

## 五、面试高频题

1. **LeetCode 46/47：** 全排列 I/II
2. **LeetCode 39/40：** 组合总和 I/II
3. **LeetCode 78/90：** 子集 I/II
4. **LeetCode 51/N皇后**
5. **LeetCode 37：** 解数独
6. **LeetCode 79：** 单词搜索
7. **LeetCode 131：** 分割回文串
8. **LeetCode 93：** 复原IP地址
9. **LeetCode 22：** 括号生成
10. **LeetCode 17：** 电话号码的字母组合
