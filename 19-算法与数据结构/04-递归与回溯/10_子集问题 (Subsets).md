# Subsets


```javascript
求集合的所有子集/幂集，是回溯的入门问题。```

## 概念说明

给定一个不含重复元素的数组，返回该数组所有可能的子集（幂集）。对于 n 个元素的集合，共有 2^n 个子集。子集问题本质上是回溯框架的应用——每个元素只有"选"或"不选"两种状态。

## 核心思路

采用增量构造法：从第一个元素开始，依次决定每个元素是否加入当前子集。递归遍历过程中，每次将当前路径（path）加入结果集，然后继续向后探索。关键在于使用 `start` 参数保证不回头取元素，避免产生重复子集。

## 复杂度分析

- **时间复杂度：** O(n * 2^n)，共 2^n 个子集，每个子集最多拷贝 n 个元素。
- **空间复杂度：** O(n)，递归栈深度为 n（不含结果存储）。

## 适用场景

- 求所有可能组合的枚举问题
- 密码破解、决策树分析
- 数据挖掘中的特征子集选择

```
function subsets(nums) {
  const res = [];
  function backtrack(start, path) {
    res.push([...path]);
    for (let i = start; i < nums.length; i++) {
      path.push(nums[i]);
      backtrack(i + 1, path);
      path.pop();
    }
  }
  backtrack(0, []);
  return res;
}
console.log(subsets([1,2,3]));
// [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]```


## 常见变体与技巧

- **含重复元素的子集：** 先排序数组，在回溯中加入 `if (i > start && nums[i] === nums[i-1]) continue` 去重。
- **子集大小限定：** 限制 path.length 在某个范围内，可加提前返回的剪枝。
- **位运算法：** 遍历 0 到 2^n - 1 的每个整数，用二进制位表示选/不选，生成子集。

## 位运算法实现

```javascript
// 位运算法生成子集
function subsetsBitwise(nums) {
  const n = nums.length;
  const res = [];
  for (let mask = 0; mask < (1 << n); mask++) {
    const subset = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) subset.push(nums[i]);
    }
    res.push(subset);
  }
  return res;
}
// 与回溯法结果相同，但实现更简洁
// 适合 n <= 20 的情况
```

## 回溯框架总结

```javascript
// 通用回溯模板
function backtrack(candidates, start, path, res) {
  res.push([...path]);          // 收集结果
  for (let i = start; i < candidates.length; i++) {
    path.push(candidates[i]);   // 做选择
    backtrack(candidates, i + 1, path, res);  // 递归
    path.pop();                 // 撤销选择
  }
}
```

  点击按钮查看结果
