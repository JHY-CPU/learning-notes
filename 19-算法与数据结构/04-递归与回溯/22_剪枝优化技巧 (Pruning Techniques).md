# 23-剪枝优化技巧 (Pruning Techniques)

剪枝是回溯算法的关键优化，通过提前排除不可能的分支加速搜索。

## 五种主要剪枝策略

| 策略 | 说明 | 示例 |
|------|------|------|
| 可行性剪枝 | 当前路径无法满足条件时终止 | sum > target |
| 最优性剪枝 | 当前路径不可能优于已知最优 | 当前步数 > best |
| 重复性剪枝 | 跳过相同元素的重复选择 | 排序后跳过相邻相同 |
| 对称性剪枝 | 利用对称性减少搜索 | N 皇后第一行只遍历一半 |
| 启发式剪枝 | 优先探索更有希望的分支 | MRV（最少剩余值） |

## JavaScript 实现

```javascript
// 1. 可行性剪枝：组合总和
function combinationSumPruned(candidates, target) {
  candidates.sort((a, b) => a - b);
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    for (let i = start; i < candidates.length; i++) {
      if (sum + candidates[i] > target) break; // 可行性剪枝
      path.push(candidates[i]);
      backtrack(i, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}

// 2. 重复性剪枝：含重复元素的子集
function subsetsWithDup(nums) {
  nums.sort((a, b) => a - b);
  const res = [];
  function backtrack(start, path) {
    res.push([...path]);
    for (let i = start; i < nums.length; i++) {
      if (i > start && nums[i] === nums[i - 1]) continue; // 重复性剪枝
      path.push(nums[i]);
      backtrack(i + 1, path);
      path.pop();
    }
  }
  backtrack(0, []);
  return res;
}

// 3. 最优性剪枝：TSP 回溯
function tspPruned(dist) {
  const n = dist.length;
  let best = Infinity;
  function backtrack(city, visited, cost, count) {
    if (cost >= best) return; // 最优性剪枝
    if (count === n) { best = Math.min(best, cost + dist[city][0]); return; }
    for (let next = 0; next < n; next++) {
      if (visited[next]) continue;
      visited[next] = true;
      backtrack(next, visited, cost + dist[city][next], count + 1);
      visited[next] = false;
    }
  }
  const visited = new Array(n).fill(false);
  visited[0] = true;
  backtrack(0, visited, 0, 1);
  return best;
}

// 4. 对称性剪枝：N 皇后
function nQueensPruned(n) {
  let count = 0;
  const cols = new Set(), diag1 = new Set(), diag2 = new Set();
  function backtrack(row) {
    if (row === n) { count++; return; }
    for (let col = 0; col < n; col++) {
      if (cols.has(col) || diag1.has(row - col) || diag2.has(row + col)) continue;
      cols.add(col); diag1.add(row - col); diag2.add(row + col);
      backtrack(row + 1);
      cols.delete(col); diag1.delete(row - col); diag2.delete(row + col);
    }
  }
  backtrack(0);
  return count;
}

// 测试
console.log(combinationSumPruned([2, 3, 6, 7], 7)); // [[2,2,3],[7]]
console.log(subsetsWithDup([1, 2, 2]));               // [[],[1],[1,2],[1,2,2],[2],[2,2]]
console.log(nQueensPruned(8));                         // 92
```

## C++ 实现

```cpp
#include <vector>
#include <algorithm>
#include <unordered_set>
using namespace std;

// 剪枝的组合总和
void backtrack(vector<int>& cand, int target, int start,
               vector<int>& path, vector<vector<int>>& res, int sum) {
    if (sum == target) { res.push_back(path); return; }
    for (int i = start; i < cand.size(); i++) {
        if (sum + cand[i] > target) break; // 剪枝
        path.push_back(cand[i]);
        backtrack(cand, target, i, path, res, sum + cand[i]);
        path.pop_back();
    }
}
```

## 剪枝效果对比

| 问题 | 无剪枝 | 有剪枝 | 减少比例 |
|------|--------|--------|---------|
| 组合总和 [2,3,6,7], 7 | 2^4 = 16 次调用 | ~6 次 | 62% |
| 含重复子集 [1,2,2] | 2^3 = 8 | 6 | 25% |
| N=8 皇后 | 8! = 40320 | ~1000 | 97% |

## 常见陷阱

1. **剪枝过早**：判断条件不准确导致漏解
2. **排序后剪枝**：break 依赖排序，忘记排序会错误剪枝
3. **对称性误用**：对称性剪枝后计数需要调整
4. **最优性剪枝条件**：`>=` 还是 `>` 需要仔细分析

## 实际应用

剪枝在竞赛中至关重要。好的剪枝可以让指数级算法在时限内通过。关键是在正确的分支点做出正确的判断——剪得太多漏解，剪得太少超时。
