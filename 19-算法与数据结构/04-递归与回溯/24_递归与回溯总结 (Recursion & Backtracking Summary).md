# 25-递归与回溯总结 (Recursion & Backtracking Summary)

递归与回溯算法核心思想、应用场景和优化方法总结。

## 递归三要素

| 要素 | 说明 | 示例 |
|------|------|------|
| 终止条件 | 何时停止递归 | n <= 1 时返回 |
| 递推关系 | 大问题分解为子问题 | fib(n) = fib(n-1) + fib(n-2) |
| 返回值组合 | 子问题结果如何合并 | return left + right |

## 回溯三要素

| 要素 | 说明 |
|------|------|
| 路径 | 已做出的选择 |
| 选择列表 | 当前可做的选择 |
| 结束条件 | 到达决策树底部 |

## JavaScript 实现

```javascript
// 回溯通用模板
function backtrackTemplate(choices, path, result, condition, prune) {
  if (condition(path)) {
    result.push([...path]);
    return;
  }
  for (let i = 0; i < choices.length; i++) {
    if (prune && prune(choices[i], path)) continue; // 剪枝
    path.push(choices[i]);   // 做选择
    backtrackTemplate(choices, path, result, condition, prune);
    path.pop();              // 撤销选择
  }
}

// 问题分类速查
const problemTypes = {
  '子集': '每个元素选/不选, start 参数避免重复',
  '组合': '从 n 个中选 k 个, start 参数',
  '排列': '全排列, used 数组标记已选',
  '组合总和': '可重复选, 递归传 i 而非 i+1',
  '分割问题': '枚举分割点, 每层尝试不同长度',
  '棋盘问题': '逐行/逐格放置, 约束检查',
  '路径搜索': 'DFS + 回溯, visited 标记',
};

// 问题类型对比
console.log('=== 回溯问题分类 ===');
for (const [type, desc] of Object.entries(problemTypes)) {
  console.log(`${type}: ${desc}`);
}
```

## 经典题目速查

| 题目 | LeetCode | 核心技巧 |
|------|----------|---------|
| 子集 | 78 | start 参数 |
| 子集 II（含重复） | 90 | 排序 + 跳过重复 |
| 组合 | 77 | start 参数 |
| 组合总和 | 39 | 传 i（可重复） |
| 组合总和 II | 40 | 传 i+1 + 排序去重 |
| 全排列 | 46 | used 数组 |
| 全排列 II | 47 | 排序 + used 去重 |
| N 皇后 | 51 | 行逐放 + 集合约束 |
| 数独 | 37 | 逐格填 + 合法检查 |
| 括号生成 | 22 | open/close 计数 |
| 单词搜索 | 79 | DFS + 原地标记 |
| 分割回文串 | 131 | 枚举分割点 |
| 复原 IP 地址 | 93 | 枚举 4 段 |

## 递归 vs 迭代

| 特性 | 递归 | 迭代 |
|------|------|------|
| 代码 | 简洁直观 | 可能复杂 |
| 栈空间 | O(深度) | 可控 |
| 栈溢出风险 | 有 | 无 |
| 调试 | 较难 | 较易 |
| 性能 | 函数调用开销 | 通常更快 |

## 复杂度总结

| 问题类型 | 典型复杂度 |
|---------|-----------|
| 子集 | O(n * 2^n) |
| 组合 | O(C(n,k) * k) |
| 排列 | O(n! * n) |
| N 皇后 | O(n!) 实际更少 |
| 数独 | O(9^空格) 实际更少 |

## 常见陷阱

1. **忘记回溯**：做了选择但没有撤销，导致状态污染
2. **深拷贝问题**：`result.push(path)` 应该是 `result.push([...path])`
3. **去重遗漏**：含重复元素时需要排序 + 跳过逻辑
4. **终止条件错误**：太早或太晚收集结果
5. **剪枝过度**：不正确的剪枝条件会漏掉有效解
