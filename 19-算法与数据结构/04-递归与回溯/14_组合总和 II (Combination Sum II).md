# Combination Sum II


```javascript
每个数字只能使用一次，且结果不能重复（需要去重）。```

## 概念说明

组合总和 II 与 I 的区别在于：每个数字在输入中只能使用一次，且数组可能包含重复元素。需要保证结果不重复。例如 [1,1,2] 中两个 1 是不同位置的元素，但选了第一个 1 和选了第二个 1 会产生重复结果。

## 核心思路

先对数组排序。回溯时传入 `i+1`（每个元素只用一次）。去重关键：`if (i > start && candidates[i] === candidates[i-1]) continue`，在同一层递归中跳过重复值。这与 I 的区别在于用 `i > start`（而非 `i > 0`）来判断同层重复。

## 复杂度分析

- **时间复杂度：** O(2^n * n)，每个元素选或不选，最坏情况拷贝路径。
- **空间复杂度：** O(n)，递归栈深度。

## 适用场景

- 带去重的组合枚举
- 处理含重复元素数据集的子集和问题
- 需要唯一结果集的搜索问题

```
function combinationSum2(candidates, target) {
  candidates.sort((a,b) => a-b);
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    if (sum > target) return;
    for (let i = start; i < candidates.length; i++) {
      if (i > start && candidates[i] === candidates[i-1]) continue;
      path.push(candidates[i]);
      backtrack(i + 1, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
console.log(combinationSum2([10,1,2,7,6,1,5], 8));```


## 常见变体与技巧

- **去重核心：** `i > start` 保证只在同一层递归中去重，不会误删不同层的选择。
- **DP 求解：** 类似 0-1 背包问题，用 `dp[j]` 表示和为 j 的方案数或方案集合。
- **打印方案数：** 若只需计数不需要输出具体方案，动态规划效率更高。

## 去重原理图解

```
排序后: [1, 1, 2, 5, 6, 7, 10], target=8

第一层选择:
  选第一个1 -> [1] -> 递归处理 [1,2,5,6,7,10]
    选第二个1 -> [1,1] -> 递归处理 [2,5,6,7,10]
  跳过第二个1 (i>start && nums[i]==nums[i-1]) ✅
  选2 -> [2] -> 递归处理 [5,6,7,10]
  ...

关键是 i > start：在同一层中跳过重复，但不同层中允许选择
```

## DP 实现（只计方案数）

```javascript
// 组合总和 II 的 DP 版本（只计数）
function combinationSum2DP(candidates, target) {
  candidates.sort((a, b) => a - b);
  const dp = new Array(target + 1).fill(0);
  dp[0] = 1;
  for (const c of candidates) {
    for (let j = target; j >= c; j--) {  // 逆序：0-1 背包
      dp[j] += dp[j - c];
    }
  }
  return dp[target];
}
```

  点击按钮查看结果
