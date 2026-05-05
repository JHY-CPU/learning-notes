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

# 递归转迭代

## 递归转迭代概述

任何递归算法都可以转换为迭代算法（使用循环和显式栈）。迭代通常更高效，但代码可能更复杂。

> **转换动机：**避免栈溢出、提高性能、节省内存。在嵌入式系统或栈空间有限的环境中尤为重要。

## 方法一：直接循环转换

对于尾递归或简单递归，可以直接用循环替代。

```javascript
// 递归
function sum(n) {
    if (n <= 0) return 0;
    return n + sum(n - 1);
}

// 迭代
function sumIter(n) {
    let result = 0;
    for (let i = 1; i <= n; i++) {
        result += i;
    }
    return result;
}```

## 方法二：使用显式栈

对于非尾递归，可以模拟调用栈进行转换。

```javascript
// 模拟栈帧结构
function factorialIter(n) {
    const stack = [];
    let result = 1;
    while (n > 0) {
        stack.push(n);
        n--;
    }
    while (stack.length > 0) {
        result *= stack.pop();
    }
    return result;
}```

## 交互示例：斐波那契数列

输入 n：
