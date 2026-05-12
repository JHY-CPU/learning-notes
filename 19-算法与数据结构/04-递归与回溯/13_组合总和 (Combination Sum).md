# Combination Sum


```javascript
找到所有和为 target 的组合，数字可以重复使用。```

## 概念说明

给定候选数字数组和目标值 target，找出所有使数字和等于 target 的组合。同一个数字可以无限制重复选取。注意 [2,2,3] 和 [2,3,2] 视为同一组合，需要去重。

## 核心思路

回溯时维护当前和 sum。关键区别：递归调用传入 `i` 而非 `i+1`，允许重复选取同一元素。提前剪枝：若 `sum > target` 立即返回。先对数组排序有助于剪枝和去重。

## 复杂度分析

- **时间复杂度：** O(target^N)，最坏情况下每个元素可重复选取，递归树深度约为 target/min。
- **空间复杂度：** O(target/min)，递归栈深度。

## 适用场景

- 零钱兑换的变体（输出所有方案而非最少硬币数）
- 资源分配问题
- 需要重复选择元素的组合枚举

```
function combinationSum(candidates, target) {
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    if (sum > target) return;
    for (let i = start; i < candidates.length; i++) {
      path.push(candidates[i]);
      backtrack(i, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
console.log(combinationSum([2,3,6,7], 7)); // [[2,2,3],[7]]```


## 常见变体与技巧

- **排序剪枝：** 先对 candidates 排序，`sum + candidates[i] > target` 时直接 break 跳出循环。
- **DP 代替回溯：** 用动态规划也可求解，状态转移方程类似完全背包问题。
- **组合总和 III：** 限制数字范围在 1~9 且每个数最多用一次，即为组合总和 III。

## 排序剪枝优化

```javascript
function combinationSumOpt(candidates, target) {
  candidates.sort((a, b) => a - b);  // 排序
  const res = [];
  function backtrack(start, path, sum) {
    if (sum === target) { res.push([...path]); return; }
    for (let i = start; i < candidates.length; i++) {
      if (sum + candidates[i] > target) break;  // 排序后可 break
      path.push(candidates[i]);
      backtrack(i, path, sum + candidates[i]);
      path.pop();
    }
  }
  backtrack(0, [], 0);
  return res;
}
```

## 与完全背包的关系

组合总和本质上是完全背包问题的方案枚举版本：
- 物品 = candidates，重量 = value = candidates[i]，背包容量 = target
- 每个物品可以选无限次
- 求恰好装满的方案数/方案集合

  点击按钮查看结果
