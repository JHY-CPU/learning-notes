[01-递归基础](01-%E9%80%92%E5%BD%92%E5%9F%BA%E7%A1%80%20(Recursion%20Basics).html)
[02-三要素](02-%E9%80%92%E5%BD%92%E4%B8%89%E8%A6%81%E7%B4%A0%20(Three%20Elements).html)
[03-调用栈](03-%E9%80%92%E5%BD%92%E8%B0%83%E7%94%A8%E6%A0%88%20(Recursion%20Call%20Stack).html)
[04-尾递归](04-%E5%B0%BE%E9%80%92%E5%BD%92%E4%BC%98%E5%8C%96%20(Tail%20Recursion).html)
[05-转迭代](05-%E9%80%92%E5%BD%92%E8%BD%AC%E8%BF%AD%E4%BB%A3%20(Recursion%20to%20Iteration).html)
[06-递推关系](06-%E9%80%92%E6%8E%A8%E5%85%B3%E7%B3%BB%E5%BB%BA%E6%A8%A1%20(Recurrence%20Modeling).html)
[07-递归树](07-%E9%80%92%E5%BD%92%E6%A0%91%E5%88%86%E6%9E%90%20(Recursion%20Tree).html)
[08-回溯基础](08-%E5%9B%9E%E6%BA%AF%E7%AE%97%E6%B3%95%E5%9F%BA%E7%A1%80%20(Backtracking%20Basics).html)
[09-回溯框架](09-%E5%9B%9E%E6%BA%AF%E4%B8%89%E8%A6%81%E7%B4%A0%20(Backtracking%20Framework).html)
[10-剪枝](10-%E5%89%AA%E6%9E%9D%E4%BC%98%E5%8C%96%20(Pruning).html)
[11-子集](11-%E5%AD%90%E9%9B%86%E9%97%AE%E9%A2%98%20(Subsets).html)
[12-组合](12-%E7%BB%84%E5%90%88%E9%97%AE%E9%A2%98%20(Combinations).html)
[13-排列](13-%E6%8E%92%E5%88%97%E9%97%AE%E9%A2%98%20(Permutations).html)
[14-组合总和](14-%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%20(Combination%20Sum).html)
[15-N皇后](15-N%E7%9A%87%E5%90%8E%E9%97%AE%E9%A2%98%20(N-Queens).html)
[16-数独](16-%E6%95%B0%E7%8B%AC%E6%B1%82%E8%A7%A3%20(Sudoku%20Solver).html)
[17-括号](17-%E6%8B%AC%E5%8F%B7%E7%94%9F%E6%88%90%20(Generate%20Parentheses).html)
[18-单词搜索](18-%E5%8D%95%E8%AF%8D%E6%90%9C%E7%B4%A2%20(Word%20Search).html)
[19-回文串](19-%E5%88%86%E5%89%B2%E5%9B%9E%E6%96%87%E4%B8%B2%20(Palindrome%20Partitioning).html)
[20-IP地址](20-%E5%A4%8D%E5%8E%9FIP%E5%9C%B0%E5%9D%80%20(Restore%20IP%20Addresses).html)
[21-子集树](21-%E5%AD%90%E9%9B%86%E6%A0%91%E4%B8%8E%E6%8E%92%E5%88%97%E6%A0%91%20(Subset%20%26%20Permutation%20Tree).html)
[22-分支限界](22-%E5%88%86%E6%94%AF%E9%99%90%E7%95%8C%E6%B3%95%20(Branch%20and%20Bound).html)
[23-迭代加深](23-%E8%BF%AD%E4%BB%A3%E5%8A%A0%E6%B7%B1%E6%90%9C%E7%B4%A2%20(Iterative%20Deepening%20DFS).html)
[24-双向搜索](24-%E5%8F%8C%E5%90%91%E6%90%9C%E7%B4%A2%20(Bidirectional%20Search).html)
[25-综合练习](25-%E9%80%92%E5%BD%92%E4%B8%8E%E5%9B%9E%E6%BA%AF%E7%BB%BC%E5%90%88%E7%BB%83%E4%B9%A0%20(Recursion%20%26%20Backtracking%20Practice).html)

# 递归三要素

## 递归三要素概述

编写正确的递归函数需要把握三个关键要素：

要素一：基本情况 (Base Case)

要素二：递归步骤 (Recursive Step)

要素三：组合结果 (Combine Results)

## 要素一：基本情况（Base Case）

递归必须包含一个或多个基本情况，用于直接返回结果而不进行递归调用。基本情况是递归的终止条件。

```javascript
// 斐波那契数列的基本情况
function fib(n) {
    if (n === 0) return 0;  // 基本情况 1
    if (n === 1) return 1;  // 基本情况 2
    return fib(n-1) + fib(n-2);  // 递归步骤
}```

> 没有基本情况的递归会导致**无限递归**，最终造成栈溢出。

## 要素二：递归步骤（Recursive Step）

将原问题分解为更小的子问题，并调用自身来解决这些子问题。每次递归调用必须向着基本情况靠近。

> **关键：**每次递归调用都必须改变状态（通常是减小问题规模），并逐步逼近基本情况。

## 要素三：将子问题的结果组合

将子问题的结果通过某种方式组合起来，形成原问题的答案。

```javascript
// 计算数组和：组合子问题结果
function sum(arr, n) {
    if (n <= 0) return 0;          // 基本情况
    return sum(arr, n-1) + arr[n-1]; // 递归 + 组合
}```

## 三要素检查器

输入一个正整数，使用递归计算其各位数字之和：

输入数字：
