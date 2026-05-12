# Combinations


```javascript
从 n 个数中选 k 个的所有组合。```

## 概念说明

给定两个整数 n 和 k，返回 1..n 中所有可能的 k 个数的组合。组合与排列的区别在于组合不考虑顺序，即 [1,2] 和 [2,1] 被视为同一组合。因此需要保证选择时只向后取，避免重复。

## 核心思路

使用回溯框架，维护 `start` 参数保证递增选取。当 path 长度等于 k 时收集结果。剪枝优化：如果剩余可选元素不足凑齐 k 个，提前终止递归。即 `path.length + (n - i + 1) < k` 时可跳过。

## 复杂度分析

- **时间复杂度：** O(C(n,k) * k)，C(n,k) 个组合，每个拷贝 k 个元素。
- **空间复杂度：** O(k)，递归栈深度为 k。

## 适用场景

- 组合数学问题（彩票、抽奖）
- 特征选择、子集筛选
- 任何需要从集合中选取固定数量元素的场景

```
function combine(n, k) {
  const res = [];
  function backtrack(start, path) {
    if (path.length === k) { res.push([...path]); return; }
    for (let i = start; i <= n; i++) {
      path.push(i);
      backtrack(i + 1, path);
      path.pop();
    }
  }
  backtrack(1, []);
  return res;
}
console.log(combine(4, 2));
// [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]```


## 常见变体与技巧

- **剪枝优化：** `if (path.length + (n - i + 1) < k) return` 提前终止，减少无效递归。
- **含重复元素：** 先排序，同层去重用 `if (i > start && nums[i] === nums[i-1]) continue`。
- **组合总和问题：** 将 k 个数的限制改为目标和的限制，即为组合总和问题。

## C++ 实现

```cpp
#include <vector>
using namespace std;

void backtrack(int n, int k, int start, vector<int>& path, vector<vector<int>>& res) {
    if (path.size() == k) { res.push_back(path); return; }
    for (int i = start; i <= n; i++) {
        // 剪枝：剩余不足
        if (path.size() + (n - i + 1) < k) return;
        path.push_back(i);
        backtrack(n, k, i + 1, path, res);
        path.pop_back();
    }
}
```

## LeetCode 相关题目

- 77. 组合
- 216. 组合总和 III
- 39. 组合总和
- 40. 组合总和 II
- 78. 子集（组合的变体）

  点击按钮查看结果
