# Permutations


```javascript
给定不含重复数字的数组，返回所有可能的全排列。```

## 概念说明

排列是指将 n 个元素按不同顺序排列，共产生 n! 种排列方式。与组合不同，排列关注顺序，[1,2] 和 [2,1] 是两个不同的排列。排列问题与子集问题的关键区别在于排列需要访问所有元素，不存在"跳过"的概念。

## 核心思路

使用 `used` 数组标记哪些元素已被选取。每次递归遍历整个数组，跳过已使用的元素。选择一个元素后标记 used，递归完成后撤销标记。与子集/组合不同，排列从 i=0 开始遍历（不使用 start），因为每个位置都可以放任何未用元素。

## 复杂度分析

- **时间复杂度：** O(n! * n)，n! 个排列，每个排列拷贝 n 个元素。
- **空间复杂度：** O(n)，递归栈深度 + used 数组。

## 适用场景

- 枚举所有排列（暴力搜索）
- 旅行商问题（TSP）的回溯求解
- 排列型动态规划的前置知识

```
function permute(nums) {
  const res = [];
  function backtrack(path, used) {
    if (path.length === nums.length) { res.push([...path]); return; }
    for (let i = 0; i < nums.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      path.push(nums[i]);
      backtrack(path, used);
      path.pop();
      used[i] = false;
    }
  }
  backtrack([], new Array(nums.length).fill(false));
  return res;
}
console.log(permute([1,2,3])); // 6个排列```


## 常见变体与技巧

- **含重复元素的排列：** 先排序，加入 `if (i > 0 && nums[i] === nums[i-1] && !used[i-1]) continue` 去重。
- **交换法：** 不用 used 数组，通过交换当前位置与后续位置实现排列，空间更优。
- **字典序下一个排列：** 若只需按字典序输出，可使用迭代法求下一个排列。

## 交换法实现

```javascript
// 交换法排列：不需要 used 数组
function permuteSwap(nums) {
  const res = [];
  function backtrack(start) {
    if (start === nums.length) { res.push([...nums]); return; }
    for (let i = start; i < nums.length; i++) {
      [nums[start], nums[i]] = [nums[i], nums[start]];  // 交换
      backtrack(start + 1);
      [nums[start], nums[i]] = [nums[i], nums[start]];  // 还原
    }
  }
  backtrack(0);
  return res;
}
```

## C++ 实现

```cpp
void permute(vector<int>& nums, vector<bool>& used,
             vector<int>& path, vector<vector<int>>& res) {
    if (path.size() == nums.size()) { res.push_back(path); return; }
    for (int i = 0; i < nums.size(); i++) {
        if (used[i]) continue;
        used[i] = true;
        path.push_back(nums[i]);
        permute(nums, used, path, res);
        path.pop_back();
        used[i] = false;
    }
}
```

  点击按钮查看结果
